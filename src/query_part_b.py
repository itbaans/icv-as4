import argparse
import cv2
import matplotlib.pyplot as plt
import os

import config
from feature_extractors import extract_hog, extract_color_hist, extract_lbp, extract_sift_bovw, extract_color_sift_bovw
from retrieval import load_index, retrieve

def main():
    parser = argparse.ArgumentParser(description="Part B: Bounding Box Object Retrieval")
    parser.add_argument("--dataset", required=True, choices=["food-101", "paris6k"], help="Which dataset index to search")
    parser.add_argument("--feature", required=True, choices=["hog", "color_hist", "lbp", "sift", "color_sift"], help="Which feature method to use")
    parser.add_argument("--query", required=True, help="Path to the query image")
    args = parser.parse_args()
    
    try:
        index_features, index_paths, index_labels, vocab_model = load_index(args.dataset, args.feature)
    except FileNotFoundError as e:
        print(e)
        return
        
    if args.feature in ["color_hist", "lbp"]:
        metric = "chi2"
    elif args.feature in ["sift", "color_sift"]:
        metric = "cosine"
    else:
        metric = "euclidean"
        
    query_img = cv2.imread(args.query)
    if query_img is None:
        print("Error: Could not read the query image.")
        return
        
    # Interactive Bounding Box Selection
    print("Please draw a bounding box around the object of interest.")
    print("Press ENTER or SPACE to confirm the selection, or 'c' to cancel.")
    
    # Needs to fit screen if image is huge
    h, w = query_img.shape[:2]
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display_img = cv2.resize(query_img, (int(w*scale), int(h*scale)))
    else:
        display_img = query_img.copy()

    bbox = cv2.selectROI(f"Select ROI on {args.dataset}", display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # If the user canceled (width or height is 0)
    if bbox[2] == 0 or bbox[3] == 0:
        print("Selection canceled by user.")
        return
        
    # Scale bounding box back to original coordinates
    x, y, bw, bh = [int(v / scale) for v in bbox]
    crop_img = query_img[y:y+bh, x:x+bw]
    
    print(f"Extracted crop of size {crop_img.shape}. Extracting {args.feature} features...")
    
    if args.feature == "hog":
        q_feat = extract_hog(crop_img)
    elif args.feature == "color_hist":
        q_feat = extract_color_hist(crop_img)
    elif args.feature == "lbp":
        q_feat = extract_lbp(crop_img)
    elif args.feature == "sift":
        q_feat = extract_sift_bovw(crop_img, vocab_model)
    elif args.feature == "color_sift":
        q_feat = extract_color_sift_bovw(crop_img, vocab_model)
        
    print(f"Retrieving top {config.TOP_K} matching full images...")
    top_indices, top_distances = retrieve(q_feat, index_features, metric=metric, top_k=config.TOP_K)
    
    # Plotting
    fig = plt.figure(figsize=(15, 6))
    fig.canvas.manager.set_window_title(f"Part B ROI CBIR - {args.feature} on {args.dataset}")
    
    # Plot full image with bounding box
    ax1 = fig.add_subplot(2, 6, 1)
    img_with_box = query_img.copy()
    cv2.rectangle(img_with_box, (x, y), (x+bw, y+bh), (0, 0, 255), max(2, int(2/scale)))
    ax1.imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Query", fontweight="bold")
    ax1.axis("off")
    
    # Plot specifically the cropped region
    ax2 = fig.add_subplot(2, 6, 2)
    ax2.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Cropped ROI", fontweight="bold", color="red")
    ax2.axis("off")
    
    for i, (idx, dist) in enumerate(zip(top_indices, top_distances)):
        path = index_paths[idx]
        label = index_labels[idx]
        
        res_img = cv2.imread(path)
        if res_img is not None:
            # i+3 to skip the gap after the 2 query slots
            ax = fig.add_subplot(2, 6, i+3)
            ax.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"#{i+1}. {label}\nDist: {dist:.2f}", fontsize=9)
            ax.axis("off")
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
