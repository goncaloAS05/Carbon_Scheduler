import datetime as dt
import os
import re
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2026, 5, 1),
}

WORKSPACE = "/tmp/airflow_text_workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# Helper function to load text if input file doesn't exist
def get_raw_input_text():
    input_file = "/home/gonca/airflow/dags/_inputs/shakespeare.txt"
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Fallback dummy corpus matching the general text size structure
        return ("The quick brown fox jumps over the lazy dog. " * 3000)

with DAG(
    dag_id='workflow_text_analysis_real',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    # --- 1. INITIAL FAN-OUT (32 Tasks): Word Count Chunks ---
    def word_count_chunk_task(chunk_id, **kwargs):
        text = get_raw_input_text()
        words = text.split()
        chunk_size = 200
        start = chunk_id * chunk_size
        end = (chunk_id + 1) * chunk_size
        
        count = len(words[start:end])
        return count

    word_count_tasks = []
    for i in range(32):
        t = PythonOperator(
            task_id=f'word_count_chunk_{i}',
            python_callable=word_count_chunk_task,
            op_kwargs={'chunk_id': i},
            params={"dur": 1, "cores": 1} # Shared with scheduling plugin
        )
        word_count_tasks.append(t)

    # --- 2. FAN-IN: Merge Word Counts ---
    def merge_word_counts_task(**kwargs):
        ti = kwargs['ti']
        counts = []
        for i in range(32):
            counts.append(ti.xcom_pull(task_ids=f'word_count_chunk_{i}'))
            
        counts_arr = np.array(counts, dtype=np.int64)
        total = int(counts_arr.sum())
        return total

    merge_wc = PythonOperator(
        task_id='merge_word_counts',
        python_callable=merge_word_counts_task,
        params={"dur": 1, "cores": 1}
    )

    # --- 3. MIDDLE STREAM: Segment Creation & Text Statistics ---
    def create_text_segments_task(**kwargs):
        text = get_raw_input_text()
        words = text.split()
        n = len(words)
        segment_size = n // 16
        
        segments = []
        for i in range(16):
            start_idx = i * segment_size
            end_idx = n if i == 15 else (i + 1) * segment_size
            segments.append(" ".join(words[start_idx:end_idx]))
        return segments

    create_segments = PythonOperator(
        task_id='create_text_segments',
        python_callable=create_text_segments_task,
        params={"dur": 1, "cores": 1}
    )

    def compute_text_statistics_task(**kwargs):
        text = get_raw_input_text()
        words = text.split()
        clean_words = [w.lower().strip('.,!?;:"()') for w in words]
        
        if clean_words:
            lengths = np.fromiter((len(w) for w in clean_words), dtype=np.int32)
            avg_word_length = float(lengths.mean())
        else:
            avg_word_length = 0.0

        sentence_count = max(1, text.count('.'))
        avg_sentence_length = len(words) / sentence_count
        simple_readability = avg_word_length + (avg_sentence_length / 10.0)

        filtered = [w for w in clean_words if len(w) > 4]
        if filtered:
            arr = np.array(filtered)
            uniques, counts = np.unique(arr, return_counts=True)
            top_idx = np.argsort(counts)[-10:][::-1]
            most_common = [[str(uniques[idx]), int(counts[idx])] for idx in top_idx]
            unique_word_count = int(uniques.size)
        else:
            most_common = []
            unique_word_count = 0

        return {
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "simple_readability_score": simple_readability,
            "most_common_words": most_common,
            "unique_word_count": unique_word_count
        }

    text_statistics = PythonOperator(
        task_id='compute_text_statistics',
        python_callable=compute_text_statistics_task,
        params={"dur": 2, "cores": 2}
    )

    # --- 4. SECOND LARGE FAN-OUT (20 Concurrent Tasks) ---
    # 4a. Segment Analyzers (16 Tasks)
    def analyze_segment_task(segment_id, **kwargs):
        ti = kwargs['ti']
        segments = ti.xcom_pull(task_ids='create_text_segments')
        text = segments[segment_id]
        words = text.split()
        
        if words:
            lengths = np.fromiter((len(w.strip('.,!?;:"()')) for w in words), dtype=np.int32)
            avg_word_length = float(lengths.mean()) if lengths.size else 0.0
        else:
            avg_word_length = 0.0
            
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        clean_words = [w.lower().strip('.,!?;:"()') for w in words]
        unique_words = int(np.unique(np.array(clean_words)).size) if clean_words else 0

        return {
            "segment_id": segment_id,
            "word_count": len(words),
            "avg_word_length": avg_word_length,
            "sentence_count": len(sentences),
            "unique_words": unique_words
        }

    segment_analysis_tasks = []
    for i in range(16):
        sat = PythonOperator(
            task_id=f'analyze_segment_{i}',
            python_callable=analyze_segment_task,
            op_kwargs={'segment_id': i},
            params={"dur": 1, "cores": 1}
        )
        segment_analysis_tasks.append(sat)

    # 4b. Core Text Extractors (4 Tasks)
    def extract_overall_keywords_task(**kwargs):
        ti = kwargs['ti']
        segments = ti.xcom_pull(task_ids='create_text_segments')
        stats = ti.xcom_pull(task_ids='compute_text_statistics')
        
        words_list = []
        for seg in segments[:50]: # Cap sample size
            words_list.extend(seg.lower().split()[:100])
            
        clean = np.array([w.strip('.,!?;:"()') for w in words_list])
        common_words = {w[0] for w in stats.get("most_common_words", [])}
        
        if clean.size:
            uniques, counts = np.unique(clean, return_counts=True)
            keep_mask = np.array([u not in common_words and len(u) > 5 for u in uniques])
            uniques, counts = uniques[keep_mask], counts[keep_mask]
            
            top_idx = np.argsort(counts)[-10:][::-1]
            top = [[str(uniques[idx]), int(counts[idx])] for idx in top_idx]
            return {"top_keywords": top, "total_keywords": int(uniques.size)}
        return {"top_keywords": [], "total_keywords": 0}

    overall_keywords = PythonOperator(
        task_id='extract_overall_keywords', python_callable=extract_overall_keywords_task, params={"dur": 1, "cores": 1}
    )

    def analyze_overall_punctuation_task(**kwargs):
        ti = kwargs['ti']
        segments = ti.xcom_pull(task_ids='create_text_segments')
        all_text = " ".join(segments)
        b = np.frombuffer(all_text.encode('utf-8'), dtype=np.uint8)
        
        punctuation_counts = {
            "periods": int((b == ord('.')).sum()),
            "commas": int((b == ord(',')).sum()),
            "exclamations": int((b == ord('!')).sum()),
            "questions": int((b == ord('?')).sum())
        }
        return {"punctuation_counts": punctuation_counts, "total_punctuation": sum(punctuation_counts.values())}

    overall_punctuation = PythonOperator(
        task_id='analyze_overall_punctuation', python_callable=analyze_overall_punctuation_task, params={"dur": 1, "cores": 1}
    )

    def calculate_overall_readability_task(**kwargs):
        ti = kwargs['ti']
        stats = ti.xcom_pull(task_ids='compute_text_statistics')
        enhanced_score = stats.get("simple_readability_score", 0.0) + (stats.get("avg_sentence_length", 0.0) * 0.1)
        return {
            "enhanced_readability_score": enhanced_score,
            "complexity_level": "simple" if enhanced_score < 8 else "medium" if enhanced_score < 12 else "complex"
        }

    overall_readability = PythonOperator(
        task_id='calculate_overall_readability', python_callable=calculate_overall_readability_task, params={"dur": 1, "cores": 1}
    )

    def detect_overall_patterns_task(**kwargs):
        ti = kwargs['ti']
        stats = ti.xcom_pull(task_ids='compute_text_statistics')
        return {"vocabulary_richness": stats.get("unique_word_count", 0) / max(1, stats.get("sentence_count", 1))}

    overall_patterns = PythonOperator(
        task_id='detect_overall_patterns', python_callable=detect_overall_patterns_task, params={"dur": 1, "cores": 1}
    )

    # --- 5. FAN-IN: Merge All 20 Analyses ---
    def merge_segment_analyses_task(**kwargs):
        ti = kwargs['ti']
        
        # Gather all 16 parallel segment metrics
        segment_results = [ti.xcom_pull(task_ids=f'analyze_segment_{idx}') for idx in range(16)]
        
        # Gather the 4 special extractors
        keywords = ti.xcom_pull(task_ids='extract_overall_keywords')
        punctuation = ti.xcom_pull(task_ids='analyze_overall_punctuation')
        readability = ti.xcom_pull(task_ids='calculate_overall_readability')
        patterns = ti.xcom_pull(task_ids='detect_overall_patterns')
        
        total_words = sum(s["word_count"] for s in segment_results)
        total_sentences = sum(s["sentence_count"] for s in segment_results)
        
        return {
            "total_words": total_words,
            "total_sentences": total_sentences,
            "keywords_analysis": keywords,
            "punctuation_analysis": punctuation,
            "readability_analysis": readability,
            "patterns_analysis": patterns
        }

    merged_analysis = PythonOperator(
        task_id='merge_segment_analyses',
        python_callable=merge_segment_analyses_task,
        params={"dur": 2, "cores": 1}
    )

    # --- 6. FINAL SPLIT TARGETS ---
    def calculate_text_metrics_task(**kwargs):
        ti = kwargs['ti']
        merged = ti.xcom_pull(task_ids='merge_segment_analyses')
        
        w_per_s = merged["total_words"] / max(1, merged["total_sentences"])
        return {"words_per_sentence": w_per_s}

    metrics = PythonOperator(
        task_id='calculate_text_metrics', python_callable=calculate_text_metrics_task, params={"dur": 1, "cores": 1}
    )

    def generate_text_summary_task(**kwargs):
        ti = kwargs['ti']
        merged = ti.xcom_pull(task_ids='merge_segment_analyses')
        return {"summary": f"Processed {merged['total_words']} words across the document corpus."}

    summary = PythonOperator(
        task_id='generate_text_summary', python_callable=generate_text_summary_task, params={"dur": 1, "cores": 1}
    )

    def final_comprehensive_report_task(**kwargs):
        ti = kwargs['ti']
        metric_res = ti.xcom_pull(task_ids='calculate_text_metrics')
        summary_res = ti.xcom_pull(task_ids='generate_text_summary')
        print(f"!!! WORKFLOW COMPLETE !!! {summary_res['summary']} | Metrics: {metric_res}")
        return {"status": "SUCCESS"}

    final_report = PythonOperator(
        task_id='final_comprehensive_report', python_callable=final_comprehensive_report_task, params={"dur": 1, "cores": 1}
    )

    # --- STRUCTURAL PIPELINE DEPENDENCIES GRAPH ---
    word_count_tasks >> merge_wc
    merge_wc >> [create_segments, text_statistics]
    
    # 20 parallel workflows bind to dependencies
    create_segments >> segment_analysis_tasks >> merged_analysis
    [create_segments, text_statistics] >> overall_keywords >> merged_analysis
    create_segments >> overall_punctuation >> merged_analysis
    text_statistics >> overall_readability >> merged_analysis
    text_statistics >> overall_patterns >> merged_analysis
    
    # Fan out to final reporting
    merged_analysis >> [metrics, summary] >> final_report