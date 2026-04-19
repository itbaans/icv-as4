import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
import config

def extract_hog(image):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, config.IMAGE_SIZE)
    features = hog(image, orientations=config.HOG_ORIENTATIONS,
                   pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                   cells_per_block=config.HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys', feature_vector=True)
    return features

def extract_color_hist(image):
    """Extract Color Histogram based on configured COLOR_SPACE."""
    image_resized = cv2.resize(image, config.IMAGE_SIZE)
    
    if config.COLOR_SPACE == "HSV":
        converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        ranges = [0, 180, 0, 256, 0, 256]
    elif config.COLOR_SPACE == "LAB":
        converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        ranges = [0, 256, 0, 256, 0, 256]
    else: # Default RGB
        converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        ranges = [0, 256, 0, 256, 0, 256]
        
    hist_color = cv2.calcHist([converted], [0, 1, 2], None, config.HIST_BINS, ranges)
    cv2.normalize(hist_color, hist_color)
    return hist_color.flatten()

def extract_lbp(image):
    """Extract standalone Local Binary Patterns (LBP) texture features."""
    image_resized = cv2.resize(image, config.IMAGE_SIZE)
    if len(image_resized.shape) == 3:
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_resized
        
    lbp = local_binary_pattern(gray, config.LBP_NUM_POINTS, config.LBP_RADIUS, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, config.LBP_NUM_POINTS + 3), range=(0, config.LBP_NUM_POINTS + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)  # Normalize
    return hist_lbp

def extract_sift_descriptors(image):
    """Extract raw SIFT keypoint descriptors from a single image."""
    image_resized = cv2.resize(image, config.SIFT_IMAGE_SIZE)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) if len(image_resized.shape) == 3 else image_resized
    sift = cv2.SIFT_create(nfeatures=config.SIFT_MAX_KEYPOINTS)
    _, descs = sift.detectAndCompute(gray, None)
    return descs

def build_sift_vocabulary(image_paths, vocab_size=config.SIFT_VOCAB_SIZE, sample_size=1000):
    """ Randomly sample images to build a Bag of Visual Words (BoVW) vocabulary. """
    import random
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
    all_descs = []
    
    for path in sample_paths:
        img = cv2.imread(path)
        if img is not None:
            descs = extract_sift_descriptors(img)
            if descs is not None:
                all_descs.append(descs)
                
    if not all_descs:
        raise ValueError("No SIFT descriptors found in the sampled images.")
        
    all_descs = np.vstack(all_descs)
    
    print(f"Clustering {all_descs.shape[0]} SIFT descriptors into {vocab_size} visual words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, batch_size=1000, n_init=3)
    kmeans.fit(all_descs)
    return kmeans

def extract_sift_bovw(image, kmeans_model):
    """Extract Bag of Visual Words histogram using the vocabulary."""
    descs = extract_sift_descriptors(image)
    hist = np.zeros(kmeans_model.n_clusters)
    if descs is not None:
        words = kmeans_model.predict(descs)
        for w in words:
            hist[w] += 1
        hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_sift_descriptors(image):
    """Extract Color SIFT descriptors by concatenating SIFT descriptors computed independently on 3 color channels."""
    sift = cv2.SIFT_create(nfeatures=config.SIFT_MAX_KEYPOINTS)
    image_resized = cv2.resize(image, config.SIFT_IMAGE_SIZE)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) if len(image_resized.shape) == 3 else image_resized
    keypoints = sift.detect(gray, None)
    
    if not keypoints:
        return None
        
    if len(image_resized.shape) == 3:
        if config.COLOR_SPACE_SIFT == "HSV":
            converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        elif config.COLOR_SPACE_SIFT == "LAB":
            converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
        else: # BGR (no conversion needed for splits to maintain RGB info, just extracting channels)
            converted = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
        channels = cv2.split(converted)
        descs_list = []
        for ch in channels:
            _, descs = sift.compute(ch, keypoints)
            if descs is not None:
                descs_list.append(descs)
            else:
                return None
                
        if len(descs_list) == 3:
            return np.hstack(descs_list) # (N, 384)
    else:
        # Fallback for grayscale images
        _, descs = sift.compute(gray, keypoints)
        return descs # (N, 128)
        
    return None

def build_color_sift_vocabulary(image_paths, vocab_size=config.SIFT_VOCAB_SIZE, sample_size=1000):
    import random
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
    all_descs = []
    
    for path in sample_paths:
        img = cv2.imread(path)
        if img is not None:
            descs = extract_color_sift_descriptors(img)
            if descs is not None and descs.shape[1] == 384: # only keep colored robust features
                all_descs.append(descs)
                
    if not all_descs:
        raise ValueError("No Color SIFT descriptors found.")
        
    all_descs = np.vstack(all_descs)
    
    print(f"Clustering {all_descs.shape[0]} Color SIFT descriptors into {vocab_size} visual words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, batch_size=1000, n_init=3)
    kmeans.fit(all_descs)
    return kmeans

def extract_color_sift_bovw(image, kmeans_model):
    descs = extract_color_sift_descriptors(image)
    hist = np.zeros(kmeans_model.n_clusters)
    if descs is not None and (descs.shape[1] == 384 or descs.shape[1] == kmeans_model.cluster_centers_.shape[1]):
        words = kmeans_model.predict(descs)
        for w in words:
            hist[w] += 1
        hist /= (hist.sum() + 1e-7)
    return hist
