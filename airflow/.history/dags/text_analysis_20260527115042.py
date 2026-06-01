import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

# Mocking the framework setup found in your project tree
# Adjust paths or replace with your local imports as needed
class DAGTaskMock:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    def compute(self, **kwargs):
        return self.func()

def DAGTask(func):
    return DAGTaskMock(func)

# --- DAG Tasks ---

@DAGTask
def split_text(text: str) -> List[str]:
    words = text.split()
    n = len(words)
    # Split the document into 4 balanced chunks for parallel processing
    chunk_size = max(1, n // 4)
    chunks = []
    for i in range(4):
        start_idx = i * chunk_size
        end_idx = n if i == 3 else (i + 1) * chunk_size
        chunks.append(" ".join(words[start_idx:end_idx]))
    return chunks

@DAGTask
def analyze_chunk(chunk_text: str, chunk_id: int) -> Dict[str, Any]:
    words = chunk_text.split()
    clean_words = [w.lower().strip('.,!?;:"()') for w in words]
    
    if clean_words:
        lengths = np.fromiter((len(w) for w in clean_words), dtype=np.int32)
        avg_len = float(lengths.mean())
    else:
        avg_len = 0.0
        
    return {
        "chunk_id": chunk_id,
        "word_count": len(words),
        "avg_word_length": avg_len
    }

@DAGTask
def compile_metrics(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate parallel chunk statistics using vector calculations
    word_counts = np.array([a["word_count"] for a in analyses], dtype=np.int64)
    avg_lengths = np.array([a["avg_word_length"] for a in analyses], dtype=np.float64)
    
    total_words = int(word_counts.sum())
    # Weighted average calculation
    overall_avg_len = float((avg_lengths * word_counts).sum() / total_words) if total_words > 0 else 0.0
    
    return {
        "total_words_processed": total_words,
        "calculated_average_word_length": overall_avg_len,
        "status": "Parallel chunk metrics successfully compiled."
    }

# --- Standalone Workflow Engine Runner Execution ---
if __name__ == "__main__":
    # Sample input string payload
    sample_text = ("The quick brown fox jumps over the lazy dog. " * 500) + " Computational linguistics optimization."
    
    start_time = time.time()
    
    # 1. Split text (Coordinator Root)
    chunks = split_text(sample_text)
    
    # 2. Parallel fan-out tasks (Simulated concurrently by scheduler)
    chunk_results = [
        analyze_chunk(chunks[0], 1),
        analyze_chunk(chunks[1], 2),
        analyze_chunk(chunks[2], 3),
        analyze_chunk(chunks[3], 4),
    ]
    
    # 3. Fan-in aggregation 
    report = compile_metrics(chunk_results)
    
    print(f"User waited: {time.time() - start_time:.4f}s")
    print("Execution Result Summary:", report)