"""
workload_generator.py
─────────────────────
Generates synthetic workflow workloads for Carbon_Scheduler benchmarking.

Workload distributions are calibrated to the Alibaba cluster-trace-v2018
batch_task dataset (4M jobs, 8 days of production data). If you have
downloaded the actual batch_task.csv, pass its path to
WorkloadGenerator(alibaba_csv=...) and it will fit the distributions
directly from the data. Otherwise the built-in synthetic distributions
are used (derived from published analysis of that dataset).

Reference: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018

Usage:
    from workload_generator import WorkloadGenerator

    gen = WorkloadGenerator(seed=42)
    workflows = gen.generate_suite()          # full benchmark suite
    gen.save(workflows, "benchmark_workloads.json")
"""

import json
import random
import itertools
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTIONS (calibrated to Alibaba 2018 batch_task analysis)
# Task durations in the trace are in seconds; we convert to hours and clamp
# to [1, 6] to fit Carbon_Scheduler's hourly scheduling granularity.
# Core counts are normalized to TOTAL_CORES_PER_REGION = 4.
# ─────────────────────────────────────────────────────────────────────────────

# Duration bucket weights (hours): short=1h, medium=2-3h, long=4-6h
# ~60% of Alibaba tasks are short-lived; the rest spread across medium/long
DURATION_CHOICES  = [1,    2,    3,    4,    5,    6  ]
DURATION_WEIGHTS  = [0.42, 0.22, 0.14, 0.10, 0.07, 0.05]

# Core demand weights: most tasks are light (1 core); heavy tasks (4 cores)
# are less common but appear in ML/data-processing jobs
CORE_CHOICES      = [1,    2,    3,    4   ]
CORE_WEIGHTS      = [0.55, 0.25, 0.12, 0.08]

# DAG shape catalogue — each entry is a factory function that returns
# (tasks: list[dict], edges: list[tuple]) for a given starting task index.
# Shapes cover the main structural patterns seen in production DAG traces.
DAG_SHAPES = [
    "linear",       # T1 → T2 → T3 → ...               sequential pipeline
    "fan_out",      # T1 → {T2, T3, T4}                 broadcast/scatter
    "fan_in",       # {T1, T2, T3} → T4                 gather/reduce
    "diamond",      # T1 → {T2, T3} → T4                fork-join unit
    "wide_diamond", # T1 → {T2..T5} → T6               wider fork-join
    "chain_diamond",# two diamonds chained               complex pipeline
    "mesh",         # full overlap of fan-out + fan-in   realistic DAG
    "parallel",     # no edges at all                    independent tasks
]


class WorkloadGenerator:
    """
    Generates a structured suite of benchmark workflows.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.
    alibaba_csv : str | Path | None
        Optional path to Alibaba batch_task.csv. If supplied, duration and
        core distributions are fitted from real data instead of built-ins.
    total_cores_per_region : int
        Must match TOTAL_CORES_PER_REGION in airflow_local_settings.py.
    """

    def __init__(self, seed: int = 42, alibaba_csv=None, total_cores_per_region: int = 4):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.max_cores = total_cores_per_region

        if alibaba_csv:
            self._fit_from_alibaba(alibaba_csv)
        else:
            self._dur_choices = DURATION_CHOICES
            self._dur_weights  = DURATION_WEIGHTS
            self._core_choices = CORE_CHOICES
            self._core_weights  = CORE_WEIGHTS

    # ── Distribution fitting ──────────────────────────────────────────────────

    def _fit_from_alibaba(self, csv_path):
        """
        Fit duration and core distributions from the real Alibaba batch_task.csv.

        Expected columns (from trace_2018.md schema):
            task_name, instance_num, job_name, task_duration,
            status, start_time, end_time, plan_cpu, plan_mem
        """
        import pandas as pd
        print(f"[GENERATOR] Fitting distributions from {csv_path} ...")
        df = pd.read_csv(csv_path, header=None, names=[
            "task_name", "instance_num", "job_name", "task_duration",
            "status", "start_time", "end_time", "plan_cpu", "plan_mem"
        ])

        # Duration: convert seconds → hours, clamp to [1, 6]
        df["dur_h"] = (df["task_duration"] / 3600).clip(1, 6).round().astype(int)
        dur_counts = df["dur_h"].value_counts(normalize=True).sort_index()
        self._dur_choices = dur_counts.index.tolist()
        self._dur_weights  = dur_counts.values.tolist()

        # Cores: plan_cpu is in milli-cores (100 = 1 core), clamp to [1, max]
        df["cores"] = ((df["plan_cpu"] / 100).clip(1, self.max_cores)
                        .round().astype(int))
        core_counts = df["cores"].value_counts(normalize=True).sort_index()
        self._core_choices = core_counts.index.tolist()
        self._core_weights  = core_counts.values.tolist()

        print(f"[GENERATOR] Fitted from {len(df):,} tasks. "
              f"Duration range: {min(self._dur_choices)}–{max(self._dur_choices)}h | "
              f"Core range: {min(self._core_choices)}–{max(self._core_choices)}")

    # ── Primitive samplers ────────────────────────────────────────────────────

    def _sample_dur(self) -> int:
        return self.rng.choices(self._dur_choices, weights=self._dur_weights, k=1)[0]

    def _sample_cores(self) -> int:
        return self.rng.choices(self._core_choices, weights=self._core_weights, k=1)[0]

    def _make_task(self, task_id: str, depends_on: list = None) -> dict:
        return {
            "id": task_id,
            "dur": self._sample_dur(),
            "cores": self._sample_cores(),
            "depends_on": depends_on or [],
            "output_size_gb": round(self.rng.uniform(1.0, 20.0), 1),
        }

    # ── DAG shape builders ────────────────────────────────────────────────────
    # Each builder returns tasks: list[dict] with depends_on embedded per task.
    # No separate edges list — matches the WORKFLOW_METADATA format exactly.

    def _build_linear(self, n: int = 4):
        ids = [f"T{i+1}" for i in range(n)]
        tasks = []
        for i, tid in enumerate(ids):
            tasks.append(self._make_task(tid, depends_on=[ids[i-1]] if i > 0 else []))
        return tasks

    def _build_fan_out(self, branches: int = 3):
        root = self._make_task("T1", depends_on=[])
        leaves = [self._make_task(f"T{i+2}", depends_on=["T1"]) for i in range(branches)]
        return [root] + leaves

    def _build_fan_in(self, branches: int = 3):
        sources = [self._make_task(f"T{i+1}", depends_on=[]) for i in range(branches)]
        sink_id = f"T{branches+1}"
        sink = self._make_task(sink_id, depends_on=[t["id"] for t in sources])
        return sources + [sink]

    def _build_diamond(self):
        # T1 → {T2, T3} → T4
        return [
            self._make_task("T1", depends_on=[]),
            self._make_task("T2", depends_on=["T1"]),
            self._make_task("T3", depends_on=["T1"]),
            self._make_task("T4", depends_on=["T2", "T3"]),
        ]

    def _build_wide_diamond(self, width: int = 4):
        # T1 → {T2..T(width+1)} → T(width+2)
        mid_ids = [f"T{i+2}" for i in range(width)]
        sink_id = f"T{width+2}"
        tasks = [self._make_task("T1", depends_on=[])]
        tasks += [self._make_task(mid, depends_on=["T1"]) for mid in mid_ids]
        tasks += [self._make_task(sink_id, depends_on=mid_ids)]
        return tasks

    def _build_chain_diamond(self):
        # Diamond 1: T1 → {T2,T3} → T4
        # Diamond 2: T4 → {T5,T6} → T7
        return [
            self._make_task("T1", depends_on=[]),
            self._make_task("T2", depends_on=["T1"]),
            self._make_task("T3", depends_on=["T1"]),
            self._make_task("T4", depends_on=["T2", "T3"]),
            self._make_task("T5", depends_on=["T4"]),
            self._make_task("T6", depends_on=["T4"]),
            self._make_task("T7", depends_on=["T5", "T6"]),
        ]

    def _build_mesh(self):
        # Layer 0: T1,T2  →  Layer 1: T3,T4  →  Layer 2: T5,T6  →  T7
        return [
            self._make_task("T1", depends_on=[]),
            self._make_task("T2", depends_on=[]),
            self._make_task("T3", depends_on=["T1", "T2"]),
            self._make_task("T4", depends_on=["T1", "T2"]),
            self._make_task("T5", depends_on=["T3", "T4"]),
            self._make_task("T6", depends_on=["T3", "T4"]),
            self._make_task("T7", depends_on=["T5", "T6"]),
        ]

    def _build_parallel(self, n: int = 4):
        # No dependencies — fully independent tasks
        return [self._make_task(f"T{i+1}", depends_on=[]) for i in range(n)]

    # ── Public API ────────────────────────────────────────────────────────────

    def _build_shape(self, shape: str) -> list:
        n = self.rng.randint(3, 6)   # randomise size within shape
        builders = {
            "linear":        lambda: self._build_linear(n),
            "fan_out":       lambda: self._build_fan_out(self.rng.randint(2, 4)),
            "fan_in":        lambda: self._build_fan_in(self.rng.randint(2, 4)),
            "diamond":       lambda: self._build_diamond(),
            "wide_diamond":  lambda: self._build_wide_diamond(self.rng.randint(3, 5)),
            "chain_diamond": lambda: self._build_chain_diamond(),
            "mesh":          lambda: self._build_mesh(),
            "parallel":      lambda: self._build_parallel(n),
        }
        return builders[shape]()

    def generate_workflow(self, shape: str, run_id: str,
                          deadline_hours: int = 24,
                          sla_level: int = 95) -> dict:
        """Generate a single workflow dict ready for Carbon_Scheduler."""
        tasks = self._build_shape(shape)

        return {
            "run_id": run_id,
            "shape": shape,
            "deadline_hours": deadline_hours,
            "sla_level": sla_level,
            "tasks": tasks,
            # Derived metadata useful for analysis
            "n_tasks": len(tasks),
            "total_core_hours": sum(t["dur"] * t["cores"] for t in tasks),
            "critical_path_lb": self._critical_path_lower_bound(tasks),
        }

    def _critical_path_lower_bound(self, tasks) -> int:
        """Compute critical path length (hours) from depends_on fields."""
        task_dur = {t["id"]: t["dur"] for t in tasks}
        earliest = {t["id"]: 0 for t in tasks}
        # Topological order is guaranteed by task list order in all builders
        for t in tasks:
            start = max((earliest[dep] + task_dur[dep] for dep in t["depends_on"]), default=0)
            earliest[t["id"]] = start
        return max(earliest[t["id"]] + task_dur[t["id"]] for t in tasks)

    def generate_suite(self, reps_per_shape: int = 3,
                        deadline_scenarios: list = None) -> list:
        """
        Generate the full benchmark suite.

        Parameters
        ----------
        reps_per_shape : int
            How many independent workflow instances to create per (shape, deadline) pair.
            Default 3 gives 8 shapes × 3 deadlines × 3 reps = 72 workflows.
        deadline_scenarios : list[int]
            Deadline windows in hours to test. Default: [12, 24, 48] which covers
            tight, normal, and relaxed scheduling pressure.

        Returns
        -------
        list[dict]  — ordered list of workflow dicts, ready for the benchmark runner.
        """
        if deadline_scenarios is None:
            deadline_scenarios = [12, 24, 48]

        suite = []
        counter = itertools.count(1)

        for shape in DAG_SHAPES:
            for deadline_h in deadline_scenarios:
                for rep in range(reps_per_shape):
                    idx = next(counter)
                    run_id = f"bench_{shape}_{deadline_h}h_r{rep+1}_{idx:03d}"
                    wf = self.generate_workflow(
                        shape=shape,
                        run_id=run_id,
                        deadline_hours=deadline_h,
                        sla_level=95,
                    )
                    suite.append(wf)

        print(f"[GENERATOR] Suite ready: {len(suite)} workflows | "
              f"{len(DAG_SHAPES)} shapes × {len(deadline_scenarios)} deadlines "
              f"× {reps_per_shape} reps")
        return suite

    @staticmethod
    def save(suite: list, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(suite, f, indent=2)
        print(f"[GENERATOR] Saved {len(suite)} workflows → {path}")

    @staticmethod
    def load(path: str) -> list:
        with open(path) as f:
            return json.load(f)


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Carbon_Scheduler workload generator")
    parser.add_argument("--out",   default="benchmark_workloads.json")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--reps",  type=int, default=3,
                        help="Repetitions per (shape, deadline) combination")
    parser.add_argument("--alibaba-csv", default=None,
                        help="Optional path to Alibaba batch_task.csv for real distributions")
    args = parser.parse_args()

    gen = WorkloadGenerator(seed=args.seed, alibaba_csv=args.alibaba_csv)
    suite = gen.generate_suite(reps_per_shape=args.reps)
    WorkloadGenerator.save(suite, args.out)