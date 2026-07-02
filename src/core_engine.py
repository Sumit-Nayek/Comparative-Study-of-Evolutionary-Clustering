import numpy as np

def calculate_distance_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix using NumPy broadcasting.
    Avoids nested loops to ensure zero CPU bloat in cloud systems.
    
    Parameters:
    data: Matrix of shape (n_samples, n_features)
    centroids: Matrix of shape (n_clusters, n_features)
    
    Returns:
    Distance matrix of shape (n_samples, n_clusters)
    """
    # Using the identity: (a - b)^2 = a^2 - 2ab + b^2 for calculation acceleration
    distances = np.sqrt(
        np.sum(data**2, axis=1, keepdims=True) 
        - 2 * np.dot(data, centroids.T) 
        + np.sum(centroids**2, axis=1)
    )
    return distances

def apply_winsorization(data: np.ndarray, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> np.ndarray:
    """
    Applies Winsorization thresholds to insulate our evolutionary multi-objective 
    fitness functions from outlier disruptions.
    """
    lower_bound = np.quantile(data, lower_quantile, axis=0)
    upper_bound = np.quantile(data, upper_quantile, axis=0)
    return np.clip(data, lower_bound, upper_bound)