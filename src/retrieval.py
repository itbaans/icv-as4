import os
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import config

def load_index(dataset_name, feature_type):
    """
    Loads pre-computed features, paths, and labels for a dataset.
    feature_type: 'hog', 'color_hist', 'lbp', 'sift', or 'color_sift'
    """
    features_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_{feature_type}.npy")
    paths_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_paths.npy")
    labels_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_labels.npy")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Index for {dataset_name}_{feature_type} not found at {features_path}. Run build_index.py first.")
        
    features = np.load(features_path)
    paths = np.load(paths_path)
    labels = np.load(labels_path)
    
    # Also load SIFT vocab if requested
    vocab_model = None
    if feature_type == 'sift':
        vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_sift_vocab.pkl")
        with open(vocab_path, "rb") as f:
            vocab_model = pickle.load(f)
    elif feature_type == 'color_sift':
        vocab_path = os.path.join(config.INDEX_DIR, f"{dataset_name}_color_sift_vocab.pkl")
        with open(vocab_path, "rb") as f:
            vocab_model = pickle.load(f)
            
    return features, paths, labels, vocab_model

def chi2_distance(A, B):
    """
    Compute the chi-squared distance between two matrices of histograms.
    A is (1, D), B is (N, D).
    """
    chi = 0.5 * np.sum((A - B)**2 / (A + B + 1e-10), axis=1)
    return chi

class IndexOptimizer:
    def __init__(self, index_features, metric="euclidean", mode="brute_force", pca_components=0.95):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        
        self.index_features = index_features
        self.metric = metric
        self.mode = mode
        self.pca = None
        self.nn_model = None
        
        # Scikit-learn NearestNeighbors doesn't directly support chi2 efficiently in KDTree, so we fallback to brute for chi2
        eff_metric = 'euclidean' if metric == 'euclidean' else ('cosine' if metric == 'cosine' else 'euclidean')
        
        if "pca" in mode:
            # Fit PCA (keep 95% variance)
            self.pca = PCA(n_components=pca_components)
            self.index_features = self.pca.fit_transform(self.index_features)
            
        if "kd_tree" in mode and metric != "chi2":
            self.nn_model = NearestNeighbors(n_neighbors=config.TOP_K + 1, algorithm='kd_tree', metric=eff_metric)
            self.nn_model.fit(self.index_features)
            
    def search(self, query_feature, top_k=config.TOP_K):
        q = query_feature.reshape(1, -1)
        
        if self.pca is not None:
            q = self.pca.transform(q)
            
        if self.nn_model is not None:
            distances, indices = self.nn_model.kneighbors(q, n_neighbors=top_k)
            return indices[0], distances[0]
        else:
            # Fallback to brute force
            if self.metric == "chi2":
                dists = chi2_distance(q, self.index_features)
            elif self.metric == "cosine":
                from scipy.spatial.distance import cdist
                dists = cdist(q, self.index_features, metric='cosine')[0]
            else:
                from scipy.spatial.distance import cdist
                dists = cdist(q, self.index_features, metric='euclidean')[0]
                
            top_indices = np.argsort(dists)[:top_k]
            top_distances = dists[top_indices]
            return top_indices, top_distances

def retrieve(query_feature, index_features, metric="euclidean", top_k=config.TOP_K, optimizer=None):
    if optimizer is not None:
        return optimizer.search(query_feature, top_k=top_k)
        
    # Legacy wrapper
    temp_opt = IndexOptimizer(index_features, metric=metric, mode="brute_force")
    return temp_opt.search(query_feature, top_k=top_k)
