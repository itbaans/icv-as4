import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import config
from retrieval import load_index, retrieve

# Ensure the output directory exists
OUT_DIR = os.path.join(config.BASE_DIR, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def plot_and_save(query_img, q_label, top_indices, top_distances, index_paths, index_labels, save_path, title_text=None):
    fig = plt.figure(figsize=(15, 6))
    
    # Plot Query
    ax = fig.add_subplot(2, 6, 1)
    ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Query: {q_label}" if q_label else "Query Region", fontweight="bold", color="blue")
    ax.axis("off")
    
    # Plot top 10 matches
    for i, (idx, dist) in enumerate(zip(top_indices, top_distances)):
        path = index_paths[idx]
        label = index_labels[idx]
        
        res_img = cv2.imread(path)
        if res_img is not None:
            ax = fig.add_subplot(2, 6, i+3)
            ax.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
            
            # Determine color: green if match, red if mismatch (only if we have a query label)
            color = "black"
            if q_label:
                color = "green" if label == q_label else "red"
                
            ax.set_title(f"#{i+1}. {label}\nDist: {dist:.2f}", fontsize=9, color=color)
            ax.axis("off")
            
    if title_text:
        plt.suptitle(title_text, fontsize=16)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {save_path}")

def generate_eda_figure(dataset_name, dataset_dir, save_path):
    # Just plots a 2x4 grid of random images to showcase dataset challenges
    paths = []
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            import glob
            imgs = glob.glob(os.path.join(class_dir, "*.jpg"))
            if imgs:
                paths.extend(random.sample(imgs, min(5, len(imgs))))
                
    if not paths:
        return
        
    samples = random.sample(paths, min(8, len(paths)))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = cv2.imread(samples[i])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = os.path.basename(os.path.dirname(samples[i]))
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved EDA figure: {save_path}")

if __name__ == "__main__":
    print("This script is meant to be customized once datasets are downloaded and index is built.")
    print("It provides functions to auto-generate grids for the LaTeX report.")
    
    # Example usage (you would uncomment this after indexing):
    
    # print("Generating EDA figures...")
    # generate_eda_figure("food-101", config.FOOD_101_DIR, os.path.join(OUT_DIR, "eda_food_samples.pdf"))
    # generate_eda_figure("paris6k", config.PARIS_6K_DIR, os.path.join(OUT_DIR, "eda_paris_samples.pdf"))
    
    # print("Generating placeholder queries (run after indexing)...")
    # try:
    #     f, p, l, _ = load_index("food-101", "hog")
    #     q_idx = 0
    #     top_idx, top_dist = retrieve(f[q_idx], f, metric="euclidean", top_k=10)
    #     top_idx = [idx for idx in top_idx if idx != q_idx][:10]
    #     img = cv2.imread(p[q_idx])
    #     plot_and_save(img, l[q_idx], top_idx, top_dist[:10], p, l, os.path.join(OUT_DIR, "query_food_hog_example.png"))
    # except Exception as e:
    #     print("Could not generate query examples (index might be missing):", e)
