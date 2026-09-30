# Carbon Scheduler

Carbon Scheduler is an Apache Airflow prototype for scheduling workflow tasks against lower-carbon execution windows and regions. It combines carbon-intensity data, task dependencies, CPU capacity, transfer costs, and deadlines to produce task placement plans.

The project has two main uses:

- Run the scheduler through Airflow and the example Kubernetes DAG.
- Run repeatable algorithm comparisons with generated workloads and benchmark scripts.

## How It Works

A workflow is described as tasks with durations, core requirements, dependencies, and optional output sizes. The scheduler uses hourly carbon-intensity data for configured regions and writes a manifest containing each task's start time and target region.

The main scheduler logic is in [`airflow/config/airflow_local_settings.py`](airflow/config/airflow_local_settings.py). It maintains hourly capacity in `airflow/cluster_state.json`, stores plans and heatmaps in `airflow/plans/`, and uses `airflow/waiting_room/` to collect workflows for batch scheduling.

When a task is ready to execute, the Airflow hook reads its manifest, waits until the planned start time, and sets the Kubernetes node selector `custom_region` to the selected region.

## Repository Layout

```text
.
├── airflow/
│   ├── config/
│   │   ├── airflow_local_settings.py   Scheduler and planning algorithms
│   │   ├── benchmark_runner.py         Benchmark execution and plots
│   │   ├── workload_generator.py       Synthetic DAG workload generation
│   │   └── cleanup_benchmark.py        Safe cleanup for benchmark outputs
│   ├── dags/                            Airflow DAG definitions
│   ├── config/data/                     Carbon CSV data and region configuration
│   ├── plans/                           Generated plans and heatmaps
│   ├── waiting_room/                    Queued workflow metadata
│   ├── cluster_state.json               Hourly regional CPU reservations
│   ├── airflow.cfg                      Local Airflow configuration
│   └── reset_cluster.py                 Reset local scheduler state
├── tests/                               Pytest tests
└── README.md
```

Runtime files under `airflow/plans/`, `airflow/waiting_room/`, `airflow/logs/`, and `airflow/cluster_state.json` are local experiment state. They should not be treated as source code.

## Requirements

- Linux or another Unix-like environment
- Python 3.12 is the tested local version
- Apache Airflow 3.2.2 for Airflow execution
- Kubernetes access for the Kubernetes DAG
- Carbon-intensity CSV files for the regions being scheduled

This checkout contains two environments:

- `airflow/airflow_venv`: Airflow runtime
- `.venv`: development and test tools, including pytest

## Installation

If the included environments are not available, create separate environments for Airflow and development tooling. Airflow has strict dependency constraints, so install it using the official constraints for your Python and Airflow versions.

For the included checkout:

```bash
source airflow/airflow_venv/bin/activate
airflow version
```

For tests:

```bash
source .venv/bin/activate
python -m pytest -q
```

## Carbon Data

The scheduler loads files named `log_<REGION>.csv` from `airflow/config/data/`. The expected columns include:

- `Datetime (UTC)`
- `Carbon intensity gCO₂eq/kWh (Life cycle)`

Region codes are read from `airflow/config/data/regions_123.txt`. If that file is missing or empty, the scheduler falls back to `DE`, `PL`, `PT`, and `ES`.

The data downloader is [`airflow/config/data/fetch_carbon_data.py`](airflow/config/data/fetch_carbon_data.py). It uses the Electricity Maps API and should be configured with a credential supplied through a secure environment or secret-management mechanism before use. Do not commit API tokens to the repository.

If you already have compatible CSV files, place them in `airflow/config/data/` and make sure their region names match `regions_123.txt`.

## Running Airflow

The checked-in configuration uses:

- DAG folder: `airflow/dags`
- `LocalExecutor`
- UTC as the default timezone
- Local simple authentication configured in `airflow/airflow.cfg`

From the repository root:

```bash
export AIRFLOW_HOME="$PWD/airflow"
source airflow/airflow_venv/bin/activate
airflow db migrate
airflow scheduler
```

In another terminal, start the web UI:

```bash
export AIRFLOW_HOME="$PWD/airflow"
source airflow/airflow_venv/bin/activate
airflow api-server --port 8080
```

Open <http://localhost:8080>. The example scheduler DAG is [`carbon_scheduler_k8s.py`](airflow/dags/carbon_scheduler_k8s.py). It defines a nine-task workflow with fan-out and fan-in dependencies and uses `KubernetesPodOperator`.

The DAG expects a Kubernetes cluster with nodes labeled using the region value selected by the scheduler, for example:

```text
custom_region=DE
```

The task metadata passed to a DAG run must include a deadline and workflow structure. The scheduler accepts task records shaped like this:

```json
{
  "id": "T1",
  "dur": 1,
  "cores": 2,
  "depends_on": [],
  "output_size_gb": 15.5
}
```

## Scheduling Algorithms

The algorithm selected for Airflow execution is controlled by `ACTIVE_ALGORITHM` in `airflow/config/airflow_local_settings.py`.

| ID | Name | Description |
| ---: | --- | --- |
| 3 | Spatio-temporal task scheduling | Places tasks across regions and time using dependencies, capacity, and RLE carbon buckets. |
| 4 | Batch planner | Schedules queued workflows using a batch heuristic. |
| 5 | Baseline A | Earliest feasible task placement across regions. |
| 6 | Baseline B | Local-first placement with external-region fallback. |
| 8 | Oracle | Compares workflow orderings using the task-level planner. |
| 9 | Atomic Oracle | Compares workflow orderings using the atomic planner. |
| 13 | Temporal batch baseline | Batch task-level scheduling with temporal shifting. |
| 15 | Spatial batch baseline | Batch task-level scheduling with spatial shifting. This is the current default. |

Batch modes use `BATCH_SIZE` and the waiting-room files to trigger coordination. The current default is three queued workflows and algorithm 15.

## Benchmarks

### Generate Workloads

The workload generator creates linear, fan-out, fan-in, diamond, mesh, and parallel DAGs with deadlines of 12, 24, and 48 hours.

```bash
cd airflow/config
python workload_generator.py --out benchmark_workloads.json
```

For shorter tasks and repeated workload shapes:

```bash
python workload_generator.py \
  --profile short \
  --reps 2 \
  --out benchmark_workloads_short.json
```

The optional `--alibaba-csv` argument fits task duration and core distributions from an Alibaba batch-task trace.

### Run the Benchmark Suite

```bash
cd airflow/config
python benchmark_runner.py \
  --workloads benchmark_workloads.json \
  --out-dir benchmark_results \
  --algorithms 8 9 42 13 15 \
  --load empty medium
```

Useful options:

- `--algorithms`: one or more algorithm IDs
- `--load`: background-load scenarios: `empty`, `light`, `medium`, or `heavy`
- `--max`: limit the number of workflows evaluated
- `--out-dir`: directory for metrics and plots

Benchmark output includes metric files, comparison plots, scheduler logs, and saved state snapshots. The main metrics include execution carbon, transfer carbon, total carbon, makespan, waiting time, deadline fulfillment, and success rate.

### Analyse Benchmark Results

After the benchmark finishes, run the post-processing script from the repository root:

```bash
python airflow/config/benchmark_results/analysis.py
```

This reads [`benchmark_results_total.csv`](airflow/config/benchmark_results/benchmark_results_total.csv) and writes the analysis plots to `airflow/config/benchmark_results/analysis/`. It prints the mean metrics for each algorithm and generates:

- runner-equivalent carbon boxplot, success-rate chart, makespan/deadline chart, background-load heatmap, one readable 3×3 origin-region carbon figure using a shared y-scale, and an average-carbon heatmap covering all regional carbon CSVs;
- quadrant plots for carbon versus makespan, success rate, transfer carbon, and decision time, with explicit lower/higher-is-better labels.

To use another results file or output folder:

```bash
python airflow/config/benchmark_results/analysis.py \
  --results path/to/benchmark_results_total.csv \
  --out-dir path/to/analysis
```

Permutation analysis is optional and can be expensive because it evaluates every ordering. For example, this evaluates at most five matching workflows (`5! = 120` orderings):

```bash
python airflow/config/benchmark_results/analysis.py \
  --permute \
  --workloads airflow/config/benchmark_workloads.json \
  --permute-max 5 \
  --permute-deadline 24
```

Use `--help` to see all options. Run the script without `--permute` for the normal CSV analysis; permutation mode is not required for the benchmark plots.

### Clean Benchmark Outputs

Preview matching generated files:

```bash
cd airflow/config
python cleanup_benchmark.py
```

Delete the matching benchmark files:

```bash
python cleanup_benchmark.py --confirm
```

Use `--all --confirm` only when all plans and heatmaps in the configured output directories can be removed.

## Reset Local Scheduler State

`airflow/reset_cluster.py` clears the local cluster state, waiting room, generated plan JSON files, and matrix test data while preserving PNG heatmaps.

Review its path settings before running it. The script defaults to `~/airflow`, while this repository normally uses `$PWD/airflow` as `AIRFLOW_HOME`:

```bash
python airflow/reset_cluster.py
```

## Tests

Run the test suite from the repository root using the development environment:

```bash
source .venv/bin/activate
python -m pytest -q
```

The tests cover carbon-data extraction and an important algorithm 3 behavior: supplying an in-memory cluster state forces a fresh plan instead of reusing a stale plan file.

## Configuration Notes

Important settings live near the top of [`airflow_local_settings.py`](airflow/config/airflow_local_settings.py):

- `ACTIVE_ALGORITHM`: scheduler mode used by the Airflow hook
- `BATCH_SIZE`: number of workflows required before batch coordination
- `TOTAL_CORES_PER_REGION`: per-region hourly CPU capacity
- `REGIONS_FILE`: source of configured region codes
- `DATA_DIR`: location of carbon CSV files (`airflow/config/data/`)
- `PLAN_DIR`: generated plans and heatmaps
- `WAITING_ROOM_DIR`: queued workflow metadata

The scheduler uses a fixed hourly planning model. Task durations, start times, capacity reservations, and carbon lookups are normalized to hourly slots.

## Known Limitations

- This is a local research and benchmarking prototype, not a production deployment guide.
- The scheduler state is stored in JSON and is not a transactional shared database.
- Kubernetes execution requires matching node labels and a working cluster.
- The benchmark and scheduler assume carbon data covers the requested planning window; missing values fall back to a default intensity.
- Several scripts use repository-relative or local paths, so run them from the directories shown above.
