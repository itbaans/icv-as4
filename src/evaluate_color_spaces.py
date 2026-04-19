import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import importlib

import config
from feature_extractors import extract_color_hist, build_color_sift_vocabulary, extract_color_sift_bovw
from build_index import get_image_paths_and_labels
from evaluate import evaluate_retrieval

def build_color_indices(dataset_name, dataset_dir):
    print(f"\n--- Rebuilding Color Features for {dataset_name} using {config.COLOR_SPACE} ---")
    paths, labels = get_image_paths_and_labels(dataset_dir)
    
    if not paths:
        print("No images found.")
        return
        
    print(f"Building Color SIFT vocabulary for {config.COLOR_SPACE}...")
    color_sift_vocab_model = build_color_sift_vocabulary(paths, vocab_size=config.SIFT_VOCAB_SIZE, sample_size=800)
    
    color_vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_color_sift_vocab.pkl")
    with open(color_vocab_path, "wb") as f:
        pickle.dump(color_sift_vocab_model, f)
        
    color_hist_features = []
    color_sift_features = []
    valid_paths = []
    valid_labels = []
    
    print("Extracting Color Histogram & Color SIFT features...")
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        img = cv2.imread(path)
        if img is None:
            continue
            
        try:
            c = extract_color_hist(img)
            cs = extract_color_sift_bovw(img, color_sift_vocab_model)
            
            color_hist_features.append(c)
            color_sift_features.append(cs)
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception as e:
            continue
            
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_color_hist.npy"), np.array(color_hist_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_color_sift.npy"), np.array(color_sift_features))
    
    # We also update paths & labels in case this is the first run
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_paths.npy"), np.array(valid_paths))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_labels.npy"), np.array(valid_labels))


def main():
    color_spaces = ["HSV", "LAB", "RGB"]
    datasets = [
        ("food-101", config.FOOD_101_DIR),
        ("paris6k", config.PARIS_6K_DIR)
    ]
    
    results = {}
    
    for cs in color_spaces:
        print("="*60)
        print(f"EVALUATING COLOR SPACE: {cs}")
        print("="*60)
        
        # Override config
        config.COLOR_SPACE = cs
        
        for name, d_dir in datasets:
            # 1. Build indices for color features
            build_color_indices(name, d_dir)
            
            # 2. Evaluate
            print(f"\nEvaluating {name} - color_hist in {cs}")
            map_hist = evaluate_retrieval(name, "color_hist", num_queries=150)
            
            print(f"Evaluating {name} - color_sift in {cs}")
            map_sift = evaluate_retrieval(name, "color_sift", num_queries=150)
            
            results[(cs, name, "color_hist")] = map_hist
            results[(cs, name, "color_sift")] = map_sift
            
    # Print Summary Report
    print("\n" + "="*60)
    print("FINAL COLOR SPACE IMPACT REPORT")
    print("="*60)
    print(f"{'Color Space':<15} | {'Dataset':<12} | {'Feature':<12} | {'mPrecision@K':<12}")
    print("-" * 60)
    for key, val in results.items():
        cs, name, feat = key
        val_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"{cs:<15} | {name:<12} | {feat:<12} | {val_str:<12}")

if __name__ == "__main__":
    main()
