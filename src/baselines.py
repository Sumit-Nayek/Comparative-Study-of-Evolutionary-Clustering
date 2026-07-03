import numpy as np
from sklearn.cluster import KMeans
from src.objectives import evaluate_compactness, evaluate_separation

def compute_classical_baselines(data: np.ndarray, n_clusters: int):
    """
    Computes the objective scores for standard single-objective K-Means 
    to act as a baseline landmark on our Pareto plots.
    """
    print("🤖 Computing standard K-Means baseline benchmarks...")
    
    # Run standard K-Means with smart initialization
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    
    # Calculate where this standard solution lands on our custom multi-objective metrics
    km_f1 = evaluate_compactness(data, centroids)
    km_f2 = evaluate_separation(centroids)
    
    return {"kmeans": [km_f1, km_f2]}