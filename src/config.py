import os

# Dataset Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FOOD_101_DIR = os.path.join(DATA_DIR, "food-101")
PARIS_6K_DIR = os.path.join(DATA_DIR, "paris6k")

# Indexing/Precomputed Feature Paths
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Shared Image Parameters
IMAGE_SIZE = (128, 128)  # Baseline resolution requirement for HOG/LBP/Color
SIFT_IMAGE_SIZE = (300, 300)  # A compromise dimension to keep SIFT details but speed up extraction

# HOG Parameters (Optimal based on ablation)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP Parameters (Optimal based on ablation)
LBP_NUM_POINTS = 16
LBP_RADIUS = 2

# Color Histogram Parameters (Optimal based on ablation)
COLOR_SPACE = "HSV"  # Options: "HSV", "LAB", "RGB"
HIST_BINS = (8, 8, 8)  # Bins per channel

# Color SIFT Parameters (Optimal based on ablation)
COLOR_SPACE_SIFT = "RGB"

# SIFT parameters (Optimal based on ablation)
SIFT_MAX_KEYPOINTS = 500
SIFT_VOCAB_SIZE = 100  # Number of visual words for K-Means clustering

# Number of top matches to retrieve
TOP_K = 10
