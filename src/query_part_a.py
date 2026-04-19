import argparse
import cv2
import matplotlib.pyplot as plt
import os

import config
from feature_extractors import extract_hog, extract_color_hist, extract_lbp, extract_sift_bovw, extract_color_sift_bovw
from retrieval import load_index, retrieve

def main():
    parser = argparse.ArgumentParser(description="Part A: Whole-Image Similarity Search")
    parser.add_argument("--dataset", required=True, choices=["food-101", "paris6k"], help="Which dataset index to search")
    parser.add_argument("--feature", required=True, choices=["hog", "color_hist", "lbp", "sift", "color_sift"], help="Which feature method to use")
    parser.add_argument("--query", required=True, help="Path to the query image")
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} index for feature: {args.feature}...")
    try:
        index_features, index_paths, index_labels, vocab_model = load_index(args.dataset, args.feature)
    except FileNotFoundError as e:
        print(e)
        return
        
    # Standardize distance metrics per feature type as documented in the report
    if args.feature in ["color_hist", "lbp"]:
        metric = "chi2"
    elif args.feature in ["sift", "color_sift"]:
        metric = "cosine"
    else:
        metric = "euclidean"
        
    print(f"Loading query image: {args.query}")
    query_img = cv2.imread(args.query)
    if query_img is None:
        print("Error: Could not read the query image. Ensure the path is correct.")
        return
        
    print(f"Extracting {args.feature} features from the query...")
    if args.feature == "hog":
        q_feat = extract_hog(query_img)
    elif args.feature == "color_hist":
        q_feat = extract_color_hist(query_img)
    elif args.feature == "lbp":
        q_feat = extract_lbp(query_img)
    elif args.feature == "sift":
        q_feat = extract_sift_bovw(query_img, vocab_model)
    elif args.feature == "color_sift":
        q_feat = extract_color_sift_bovw(query_img, vocab_model)
        
    print(f"Retrieving top {config.TOP_K} matches from the index using '{metric}' distance...")
    top_indices, top_distances = retrieve(q_feat, index_features, metric=metric, top_k=config.TOP_K)
    
    print("\n" + "="*50)
    print(f"Results for Query: {os.path.basename(args.query)}")
    print("="*50)
    
    # Plotting code
    fig = plt.figure(figsize=(15, 6))
    fig.canvas.manager.set_window_title(f"CBIR Results - {args.feature} on {args.dataset}")
    
    # Plot Query
    ax = fig.add_subplot(2, 6, 1)
    ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Query Image", fontweight="bold", color="blue")
    ax.axis("off")
    
    for i, (idx, dist) in enumerate(zip(top_indices, top_distances)):
        path = index_paths[idx]
        label = index_labels[idx]
        print(f"{i+1:2d}. [Dist: {dist:.4f}] Class: {label:<20} | Path: {path}")
        
        res_img = cv2.imread(path)
        if res_img is not None:
            # i+3 to skip the gap after the query image (which is at position 1)
            ax = fig.add_subplot(2, 6, i+3)
            ax.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"#{i+1}. {label}\nDist: {dist:.2f}", fontsize=9)
            ax.axis("off")
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
