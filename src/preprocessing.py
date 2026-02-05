import numpy as np

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize_features(X, mean=None, std=None):
    if mean is None or std is None:        
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    
    X_norm = (X - mean) / std
    return X_norm, mean, std