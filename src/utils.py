import numpy as np
from sklearn.datasets import make_blobs

def generate_synthetic_spaces():
    """
    Generates data arrays for our dual-space analysis strategy:
    1. Classical Space: Low dimensional, clearly defined spherical clusters.
    2. Embedding Space: High dimensional, highly overlapping layout.
    """
    # 1. Classical Space Configuration (500 samples, 2 dimensions, 3 accurate centers)
    classical_data, _ = make_blobs(
        n_samples=500, n_features=2, centers=3, cluster_std=0.60, random_state=42
    )
    
    # 2. Mock Embedding Space Configuration (300 samples, 128 dimensions, highly overlapping)
    embedding_data, _ = make_blobs(
        n_samples=300, n_features=128, centers=4, cluster_std=2.50, random_state=42
    )
    
    return classical_data, embedding_data