import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

import config
from feature_extractors import extract_hog, extract_lbp
from build_index import get_image_paths_and_labels
from retrieval import IndexOptimizer
from evaluate import evaluate_retrieval
from evaluate_color_spaces import build_color_indices

def build_specific_feature(dataset_name, dataset_dir, feature_flag):
    paths, labels = get_image_paths_and_labels(dataset_dir)
    if not paths:
        return
        
    features = []
    valid_paths = []
    valid_labels = []
    
    for path, label in zip(paths, labels):
        img = cv2.imread(path)
        if img is None:
            continue
            
        try:
            if feature_flag == "hog":
                f = extract_hog(img)
            elif feature_flag == "lbp":
                f = extract_lbp(img)
            features.append(f)
            valid_paths.append(path)
            valid_labels.append(label)
        except:
            continue
            
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_{feature_flag}.npy"), np.array(features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_paths.npy"), np.array(valid_paths))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_labels.npy"), np.array(valid_labels))


def build_sift_feature(dataset_name, dataset_dir):
    from feature_extractors import build_sift_vocabulary, extract_sift_bovw
    import pickle
    
    paths, labels = get_image_paths_and_labels(dataset_dir)
    if not paths:
        return
        
    print(f"Building SIFT vocabulary for {dataset_name} (Vocab:{config.SIFT_VOCAB_SIZE}, MaxKP:{config.SIFT_MAX_KEYPOINTS})...")
    sift_vocab_model = build_sift_vocabulary(paths, vocab_size=config.SIFT_VOCAB_SIZE, sample_size=800)
    
    vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_sift_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(sift_vocab_model, f)
        
    features = []
    valid_paths = []
    valid_labels = []
    print(f"Extracting SIFT BoVW for {dataset_name}...")
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            s = extract_sift_bovw(img, sift_vocab_model)
            features.append(s)
            valid_paths.append(path)
            valid_labels.append(label)
        except:
            continue
            
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_sift.npy"), np.array(features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_paths.npy"), np.array(valid_paths))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_labels.npy"), np.array(valid_labels))

def test_internal_ablations():
    print("\n" + "="*70)
    print("MASTER ABLATION STUDY (SECTION 5.1)")
    print("="*70)
    
    best_configs = {}
    
    # ---------------- 1. HOG ABLATIONS ----------------
    print("\n--- Phase 1: HOG Ablations ---")
    hog_configs = [
        {"orient": 6, "ppc": (16, 16)},
        {"orient": 9, "ppc": (16, 16)},
        {"orient": 6, "ppc": (8, 8)},
        {"orient": 9, "ppc": (8, 8)},
    ]
    
    best_hog_score = 0
    best_hog_cfg = None
    
    for c in hog_configs:
        config.HOG_ORIENTATIONS = c["orient"]
        config.HOG_PIXELS_PER_CELL = c["ppc"]
        print(f"\nBuilding HOG Index (Or:{c['orient']}, PPC:{c['ppc']})...")
        
        avg_score = 0
        for dataset, d_dir in [("food-101", config.FOOD_101_DIR), ("paris6k", config.PARIS_6K_DIR)]:
            build_specific_feature(dataset, d_dir, "hog")
            score = evaluate_retrieval(dataset, "hog", num_queries=200)
            avg_score += (score if score else 0)
        
        avg_score /= 2.0
        if avg_score > best_hog_score:
            best_hog_score = avg_score
            best_hog_cfg = c
            
    best_configs["hog"] = (best_hog_cfg, best_hog_score)

    # ---------------- 2. LBP ABLATIONS ----------------
    print("\n--- Phase 2: LBP Ablations ---")
    lbp_configs = [
        {"points": 8, "radius": 1},
        {"points": 16, "radius": 2},
        {"points": 24, "radius": 3},
    ]
    
    best_lbp_score = 0
    best_lbp_cfg = None
    
    for c in lbp_configs:
        config.LBP_NUM_POINTS = c["points"]
        config.LBP_RADIUS = c["radius"]
        print(f"\nBuilding LBP Index (Points:{c['points']}, Radius:{c['radius']})...")
        
        avg_score = 0
        for dataset, d_dir in [("food-101", config.FOOD_101_DIR), ("paris6k", config.PARIS_6K_DIR)]:
            build_specific_feature(dataset, d_dir, "lbp")
            score = evaluate_retrieval(dataset, "lbp", num_queries=200)
            avg_score += (score if score else 0)
            
        avg_score /= 2.0
        if avg_score > best_lbp_score:
            best_lbp_score = avg_score
            best_lbp_cfg = c
            
    best_configs["lbp"] = (best_lbp_cfg, best_lbp_score)

    # ---------------- 3. STANDARD SIFT ABLATIONS ----------------
    print("\n--- Phase 3: Standard SIFT Ablations ---")
    sift_configs = [
        {"vocab": 50, "max_kp": 250},
        {"vocab": 100, "max_kp": 500},
    ]
    
    best_sift_score = 0
    best_sift_cfg = None
    
    for c in sift_configs:
        config.SIFT_VOCAB_SIZE = c["vocab"]
        config.SIFT_MAX_KEYPOINTS = c["max_kp"]
        print(f"\nBuilding SIFT Index (Vocab:{c['vocab']}, MaxKP:{c['max_kp']})...")
        
        avg_score = 0
        for dataset, d_dir in [("food-101", config.FOOD_101_DIR), ("paris6k", config.PARIS_6K_DIR)]:
            build_sift_feature(dataset, d_dir)
            score = evaluate_retrieval(dataset, "sift", num_queries=200)
            avg_score += (score if score else 0)
            
        avg_score /= 2.0
        if avg_score > best_sift_score:
            best_sift_score = avg_score
            best_sift_cfg = c
            
    best_configs["sift"] = (best_sift_cfg, best_sift_score)

    # ---------------- 4. COLOR SPACE ABLATIONS ----------------
    print("\n--- Phase 4: Color + SIFT Space Ablations ---")
    color_spaces = ["HSV", "LAB", "RGB"]
    
    best_hist_score = 0
    best_hist_cfg = None
    
    best_csift_score = 0
    best_csift_cfg = None
    
    for cs in color_spaces:
        config.COLOR_SPACE = cs
        
        avg_hist_score = 0
        avg_csift_score = 0
        
        for name, d_dir in [("food-101", config.FOOD_101_DIR), ("paris6k", config.PARIS_6K_DIR)]:
            build_color_indices(name, d_dir)
            score_hist = evaluate_retrieval(name, "color_hist", num_queries=200)
            score_sift = evaluate_retrieval(name, "color_sift", num_queries=200)
            avg_hist_score += (score_hist if score_hist else 0)
            avg_csift_score += (score_sift if score_sift else 0)
            
        avg_hist_score /= 2.0
        avg_csift_score /= 2.0
        
        if avg_hist_score > best_hist_score:
            best_hist_score = avg_hist_score
            best_hist_cfg = cs
            
        if avg_csift_score > best_csift_score:
            best_csift_score = avg_csift_score
            best_csift_cfg = cs
            
    best_configs["color_hist"] = (best_hist_cfg, best_hist_score)
    best_configs["color_sift"] = (best_csift_cfg, best_csift_score)
    
    # ---------------- 5. FINAL COMPARISON ----------------
    report_text = "\n" + "="*80 + "\n"
    report_text += "FINAL FEATURE DESCRIPTOR COMPARISON (Best Configs Across Both Datasets)\n"
    report_text += "="*80 + "\n"
    report_text += f"{'Feature':<15} | {'Best Configuration Found':<35} | {'Avg mPrecision@10':<15}\n"
    report_text += "-" * 80 + "\n"
    
    report_text += f"HOG             | {str(best_configs['hog'][0]):<35} | {best_configs['hog'][1]:.4f}\n"
    report_text += f"LBP             | {str(best_configs['lbp'][0]):<35} | {best_configs['lbp'][1]:.4f}\n"
    report_text += f"Color Hist      | {str(best_configs['color_hist'][0]):<35} | {best_configs['color_hist'][1]:.4f}\n"
    report_text += f"Standard SIFT   | {str(best_configs['sift'][0]):<35} | {best_configs['sift'][1]:.4f}\n"
    report_text += f"Color SIFT      | {str(best_configs['color_sift'][0]):<35} | {best_configs['color_sift'][1]:.4f}\n"

    print(report_text)
    
    import os
    report_dir = os.path.join(config.BASE_DIR, "report")
    os.makedirs(report_dir, exist_ok=True)
    save_path = os.path.join(report_dir, "ablation_results.txt")
    with open(save_path, "w") as f:
        f.write(report_text)
        
    print(f"\n--> Check {save_path} for saved results.")

if __name__ == "__main__":
    test_internal_ablations()
