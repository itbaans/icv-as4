import os
import glob
import cv2
import numpy as np
import pickle
from tqdm import tqdm

import config
from feature_extractors import extract_hog, extract_color_hist, extract_lbp, build_sift_vocabulary, extract_sift_bovw, build_color_sift_vocabulary, extract_color_sift_bovw

def get_image_paths_and_labels(dataset_dir):
    paths = []
    labels = []
    # Both datasets are usually structured as dataset_dir/class_name/image.jpg
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for img_path in glob.glob(os.path.join(class_dir, "*.jpg")):
                paths.append(img_path)
                labels.append(class_name)
    return paths, labels

def process_dataset(dataset_name, dataset_dir):
    print(f"\n[{dataset_name}] Processing dataset from {dataset_dir}")
    paths, labels = get_image_paths_and_labels(dataset_dir)
    
    if not paths:
        print(f"[{dataset_name}] Skipping: No images found. Make sure data is downloaded and unzipped correctly into {dataset_dir}.")
        return
        
    print(f"[{dataset_name}] Found {len(paths)} images. Building SIFT and Color SIFT vocabularies...")
    sift_vocab_model = build_sift_vocabulary(paths)
    color_sift_vocab_model = build_color_sift_vocabulary(paths)
    
    vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_sift_vocab.pkl")
    color_vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_color_sift_vocab.pkl")
    
    with open(vocab_path, "wb") as f:
        pickle.dump(sift_vocab_model, f)
    with open(color_vocab_path, "wb") as f:
        pickle.dump(color_sift_vocab_model, f)
        
    print(f"[{dataset_name}] Extracting features (HOG, Color Hist, LBP, SIFT BoVW, Color SIFT)...")
    hog_features = []
    color_hist_features = []
    lbp_features = []
    sift_features = []
    color_sift_features = []
    valid_paths = []
    valid_labels = []
    
    # We use tqdm for a nice terminal progress bar
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        img = cv2.imread(path)
        if img is None:
            continue
            
        try:
            h = extract_hog(img)
            c = extract_color_hist(img)
            l = extract_lbp(img)
            s = extract_sift_bovw(img, sift_vocab_model)
            cs = extract_color_sift_bovw(img, color_sift_vocab_model)
            
            hog_features.append(h)
            color_hist_features.append(c)
            lbp_features.append(l)
            sift_features.append(s)
            color_sift_features.append(cs)
            
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception as e:
            print(f"[{dataset_name}] Error processing {path}: {e}")
            continue
            
    print(f"[{dataset_name}] Saving index files...")
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_hog.npy"), np.array(hog_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_color_hist.npy"), np.array(color_hist_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_lbp.npy"), np.array(lbp_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_sift.npy"), np.array(sift_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_color_sift.npy"), np.array(color_sift_features))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_paths.npy"), np.array(valid_paths))
    np.save(os.path.join(config.INDEX_DIR, f"{dataset_name}_labels.npy"), np.array(valid_labels))
    print(f"[{dataset_name}] Indexing complete.")

if __name__ == "__main__":
    process_dataset("food-101", config.FOOD_101_DIR)
    process_dataset("paris6k", config.PARIS_6K_DIR)
