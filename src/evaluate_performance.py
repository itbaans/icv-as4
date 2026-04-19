import time
import argparse
import random
import config
from retrieval import load_index, IndexOptimizer
from build_index import process_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=500, help="Number of random queries to test.")
    args = parser.parse_args()
    
    print("="*60)
    print("COMPUTATIONAL PERFORMANCE EVALUATION (Section 5.2)")
    print("="*60)
    
    # 1. TIMING INDEXING
    print("\n--- TIMING INDEXING ---")
    start_time = time.time()
    # To prevent blowing away everything, we just re-index paris6k as a benchmark
    print(f"Indexing paris6k at resolution {config.IMAGE_SIZE}...")
    try:
        process_dataset("paris6k", config.PARIS_6K_DIR)
        index_time = time.time() - start_time
        print(f"Index built completely in {index_time:.2f} seconds.")
    except Exception as e:
        print(f"Indexing benchmark skipped: {e}")
        
    # 2. TIMING RETRIEVAL MODES
    print("\n--- TIMING RETRIEVAL OVER 4 MODES ---")
    metrics = []
    
    # We test on HOG for pure distance comparisons (PCA/KDTree doesn't work well on Chi2)
    feature = "hog"
    dataset = "paris6k"
    
    print(f"Loading {dataset} index for {feature}...")
    try:
        index_features, index_paths, index_labels, vocab_model = load_index(dataset, feature)
    except Exception as e:
        print("Cannot find index. Did indexing fail?", e)
        return
        
    num_images = len(index_labels)
    query_indices = random.sample(range(num_images), min(args.subset, num_images))
    
    modes_to_test = ["brute_force", "pca_brute", "kd_tree", "pca_kd_tree"]
    
    for mode in modes_to_test:
        print(f"\nInitializing Optimizer for {mode}...")
        init_start = time.time()
        optimizer = IndexOptimizer(index_features, metric="euclidean", mode=mode, pca_components=0.95)
        init_time = time.time() - init_start
        
        # Benchmarking query phase
        total_precision = 0.0
        query_start = time.time()
        for q_idx in query_indices:
            q_feat = index_features[q_idx]
            q_label = index_labels[q_idx]
            
            top_indices, _ = optimizer.search(q_feat, top_k=config.TOP_K + 1)
            mapped_indices = [idx for idx in top_indices if idx != q_idx][:config.TOP_K]
            
            correct = sum([1 for idx in mapped_indices if index_labels[idx] == q_label])
            total_precision += (correct / config.TOP_K)
            
        q_time = (time.time() - query_start) / len(query_indices)
        mAP = total_precision / len(query_indices)
        
        metrics.append((mode, init_time, q_time, mAP))
        
    # Output Results
    report_text = "\n" + "="*70 + "\n"
    report_text += "FINAL PERFORMANCE REPORT: HOG ON PARIS6K (EUCLIDEAN)\n"
    report_text += "="*70 + "\n"
    report_text += f"{'Method':<15} | {'Setup Time (s)':<15} | {'Time per Query (s)':<20} | {'mPrecision@10':<12}\n"
    report_text += "-" * 75 + "\n"
    for m, i_time, q_time, acc in metrics:
        report_text += f"{m:<15} | {i_time:<15.4f} | {q_time:<20.4f} | {acc:.4f}\n"

    print(report_text)
    
    import os
    report_dir = os.path.join(config.BASE_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, "performance_results.txt")
    with open(save_path, "w") as f:
        f.write(report_text)
        
    print(f"\n--> Check {save_path} for saved results.")

if __name__ == "__main__":
    main()
