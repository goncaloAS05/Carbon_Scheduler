import os
import re
import numpy as np
import datetime as dt
from airflow import DAG
from airflow.decorators import task

# --- CONFIGURATION ---
DATA_DIR = "/home/gonca/airflow/data/text_analysis_test/"
os.makedirs(DATA_DIR, exist_ok=True)

# Metadata for Planner matching your structure pattern
WORKFLOW_METADATA = [
    {"id": "setup_text", "dur": 1, "cores": 1},
    {"id": "extract_mapping_args", "dur": 1, "cores": 1},
    {"id": "merge_word_counts", "dur": 1, "cores": 1}
]
for i in range(32):
    WORKFLOW_METADATA.append({"id": f"word_count_chunk__{i}", "dur": 1, "cores": 1})

WORKFLOW_METADATA.extend([
    {"id": "create_text_segments", "dur": 1, "cores": 1},
    {"id": "compute_text_statistics", "dur": 2, "cores": 2},
    {"id": "extract_overall_keywords", "dur": 1, "cores": 1},
    {"id": "analyze_overall_punctuation", "dur": 1, "cores": 1},
    {"id": "calculate_overall_readability", "dur": 1, "cores": 1},
    {"id": "detect_overall_patterns", "dur": 1, "cores": 1}
])

for i in range(16):
    WORKFLOW_METADATA.append({"id": f"analyze_segment__{i}", "dur": 1, "cores": 1})

WORKFLOW_METADATA.extend([
    {"id": "merge_segment_analyses", "dur": 2, "cores": 1},
    {"id": "calculate_text_metrics", "dur": 1, "cores": 1},
    {"id": "generate_text_summary", "dur": 1, "cores": 1},
    {"id": "final_comprehensive_report", "dur": 1, "cores": 1}
])

# --- FIXED EDGES FOR THE PLANNER ---
EDGES = []
for i in range(32):
    EDGES.append((f"word_count_chunk__{i}", "merge_word_counts"))
EDGES.extend([
    ("setup_text", "extract_mapping_args"),
    ("extract_mapping_args", "word_count_chunk"),
    ("merge_word_counts", "create_text_segments"),
    ("merge_word_counts", "compute_text_statistics"),
    ("create_text_segments", "extract_overall_keywords"),
    ("compute_text_statistics", "extract_overall_keywords"),
    ("create_text_segments", "analyze_overall_punctuation"),
    ("compute_text_statistics", "calculate_overall_readability"),
    ("compute_text_statistics", "detect_overall_patterns")
])
for i in range(16):
    EDGES.append(("create_text_segments", f"analyze_segment__{i}"))
    EDGES.append((f"analyze_segment__{i}", "merge_segment_analyses"))

EDGES.extend([
    ("extract_overall_keywords", "merge_segment_analyses"),
    ("analyze_overall_punctuation", "merge_segment_analyses"),
    ("calculate_overall_readability", "merge_segment_analyses"),
    ("detect_overall_patterns", "merge_segment_analyses"),
    ("merge_segment_analyses", "calculate_text_metrics"),
    ("merge_segment_analyses", "generate_text_summary"),
    ("calculate_text_metrics", "final_comprehensive_report"),
    ("generate_text_summary", "final_comprehensive_report")
])

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2026, 5, 5),
}

with DAG(
    dag_id="workflow_text_analysis_carbon",
    default_args=default_args,
    schedule=None,
    catchup=False,
    params={
        "deadline_iso": (dt.datetime.now() + dt.timedelta(days=3)).isoformat(),
        "workflow_structure": WORKFLOW_METADATA,
        "edges": EDGES,
        "sla_level": 95
    }
) as dag:

    @task
    def setup_text(dag_run=None):
        run_id = dag_run.run_id
        path = os.path.join(DATA_DIR, run_id)
        os.makedirs(path, exist_ok=True)
        
        # Simulated text payload for actual code execution
        corpus = ("The quick brown fox jumps over the lazy dog. " * 2000)
        f_path = os.path.join(path, "raw_corpus.txt")
        with open(f_path, "w") as f:
            f.write(corpus)
            
        # Create map targets for the downstream chunk processing loops
        chunk_args = [{"chunk_id": idx, "file_path": f_path} for idx in range(32)]
        return {"chunk_args": chunk_args, "path": path, "file_path": f_path}

    # --- FIX: Extracts key explicitly within a task context to bypass the proxy dictionary limit ---
    @task
    def extract_mapping_args(setup_output):
        return setup_output["chunk_args"]

    @task(task_id="word_count_chunk")
    def word_count_chunk(chunk_id, file_path):
        with open(file_path, "r") as f:
            words = f.read().split()
        start = chunk_id * 200
        end = (chunk_id + 1) * 200
        return len(words[start:end])

    @task
    def merge_word_counts(counts, info):
        total = int(np.array(counts).sum())
        return {"total_words": total, "info": info}

    @task
    def create_text_segments(merge_out):
        with open(merge_out["info"]["file_path"], "r") as f:
            words = f.read().split()
        n = len(words)
        segment_size = n // 16
        
        segment_paths = []
        for i in range(16):
            start_idx = i * segment_size
            end_idx = n if i == 15 else (i + 1) * segment_size
            seg_text = " ".join(words[start_idx:end_idx])
            
            p = os.path.join(merge_out["info"]["path"], f"seg_{i}.txt")
            with open(p, "w") as f:
                f.write(seg_text)
            segment_paths.append({"segment_id": i, "path": p})
        return segment_paths

    @task
    def compute_text_statistics(merge_out):
        with open(merge_out["info"]["file_path"], "r") as f:
            text = f.read()
        words = text.split()
        clean_words = [w.lower().strip('.,!') for w in words]
        lengths = np.fromiter((len(w) for w in clean_words if w), dtype=np.int32)
        
        return {
            "avg_word_length": float(lengths.mean()) if lengths.size else 0.0,
            "sentence_count": max(1, text.count('.')),
            "unique_word_count": len(set(clean_words))
        }

    @task(task_id="analyze_segment")
    def analyze_segment(segment_id, path):
        with open(path, "r") as f:
            words = f.read().split()
        return {"segment_id": segment_id, "count": len(words)}

    @task
    def extract_overall_keywords(segments, stats):
        return {"status": "extracted"}

    @task
    def analyze_overall_punctuation(segments):
        return {"puncts": 0}

    @task
    def calculate_overall_readability(stats):
        return {"score": 10.0}

    @task
    def detect_overall_patterns(stats):
        return {"diversity": "medium"}

    @task
    def merge_segment_analyses(segment_outputs, keywords, puncts, read, patterns):
        return {"merged": True}

    @task
    def calculate_text_metrics(merged_in):
        return {"complexity": 5}

    @task
    def generate_text_summary(merged_in):
        return {"summary": "Done"}

    @task(task_id="final_comprehensive_report")
    def final_comprehensive_report(metrics, summary):
        return "SUCCESS"

    # --- EXECUTION FLOW CONNECTIVITY ---
    init_data = setup_text()
    
    # Extract the argument lists cleanly via standard XCom tracking strings
    mapped_args = extract_mapping_args(init_data)
    
    # Parallel mapping groups execute smoothly
    counts = word_count_chunk.expand_kwargs(mapped_args)
    merged_wc = merge_word_counts(counts, init_data)
    
    segments = create_text_segments(merged_wc)
    stats = compute_text_statistics(merged_wc)
    
    # 16 segment instances mapped
    seg_analyses = analyze_segment.expand_kwargs(segments)
    
    kw = extract_overall_keywords(segments, stats)
    pa = analyze_overall_punctuation(segments)
    re_score = calculate_overall_readability(stats)
    pat = detect_overall_patterns(stats)
    
    merged_cube = merge_segment_analyses(seg_analyses, kw, pa, re_score, pat)
    
    m_out = calculate_text_metrics(merged_cube)
    s_out = generate_text_summary(merged_cube)
    
    final_report = final_comprehensive_report(m_out, s_out)