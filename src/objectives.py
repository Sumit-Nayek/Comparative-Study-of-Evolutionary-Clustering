import numpy as np
from src.core_engine import calculate_distance_matrix

def evaluate_compactness(data: np.ndarray, centroids: np.ndarray) -> float:
    """
    Objective f1: Minimize intra-cluster compactness variance.
    Calculates the hard partition sum of squared errors (SSE).
    """
    distances = calculate_distance_matrix(data, centroids)
    # Assign each point to its closest cluster center
    closest_cluster_idx = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(len(data)), closest_cluster_idx]
    
    # Square and sum up total distances to represent geometric compactness
    return float(np.sum(min_distances ** 2))

def evaluate_separation(centroids: np.ndarray) -> float:
    """
    Objective f2: Maximize inter-cluster separation.
    Returns the minimum Euclidean distance between any pair of cluster centers.
    We return it as a negative value because optimization algorithms naturally 
    seek to minimize functions (Minimizing -Separation == Maximizing Separation).
    """
    n_clusters = centroids.shape[0]
    if n_clusters < 2:
        return 0.0
        
    # Calculate the cross-distance matrix between all centroids
    centroid_distances = calculate_distance_matrix(centroids, centroids)
    
    # Fill diagonal entries with infinity so a centroid doesn't pick its own distance (0)
    np.fill_diagonal(centroid_distances, np.inf)
    
    min_separation = np.min(centroid_distances)
    return float(-min_separation)