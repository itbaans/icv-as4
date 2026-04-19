import argparse
import random
import config
from retrieval import load_index, retrieve

def evaluate_retrieval(dataset_name, feature_type, num_queries=200):
    # print(f"Loading {dataset_name} index for {feature_type}...")
    try:
        index_features, index_paths, index_labels, vocab_model = load_index(dataset_name, feature_type)
    except FileNotFoundError:
        print(f"Skipping {dataset_name}-{feature_type}: Index not found.")
        return None
        
    num_images = len(index_labels)
    if num_images == 0:
        print("Empty dataset.")
        return None
        
    if feature_type in ["color_hist", "lbp"]:
        metric = "chi2"
    elif feature_type in ["sift", "color_sift"]:
        metric = "cosine"
    else:
        metric = "euclidean"
        
    query_indices = random.sample(range(num_images), min(num_queries, num_images))
    total_precision = 0.0
    
    for q_idx in query_indices:
        q_feat = index_features[q_idx]
        q_label = index_labels[q_idx]
        
        # Retrieve top_k + 1, because the query image itself is in the index and will be a zero-distance match
        top_indices, _ = retrieve(q_feat, index_features, metric=metric, top_k=config.TOP_K + 1)
        
        # Filter out the self-match
        top_indices = [idx for idx in top_indices if idx != q_idx][:config.TOP_K]
        
        correct = sum([1 for idx in top_indices if index_labels[idx] == q_label])
        total_precision += (correct / config.TOP_K)
        
    mAP = total_precision / len(query_indices)
    print(f"Dataset: {dataset_name:<10} | Feature: {feature_type:<10} | Metric: {metric:<9} | mPrecision@{config.TOP_K}: {mAP:.4f}")
    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=200, help="Number of random queries to sample per dataset/feature combo")
    args = parser.parse_args()
    
    print("="*60)
    print(f"EVALUATING SYSTEM CLASSIFICATION PERFORMANCE - ({args.num_queries} queries)")
    print("="*60)
    
    for dataset in ["food-101", "paris6k"]:
        for feature in ["hog", "color_hist", "lbp", "sift", "color_sift"]:
            evaluate_retrieval(dataset, feature, num_queries=args.num_queries)
