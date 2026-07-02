import numpy as np

def calculate_distance_matrix(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix using NumPy broadcasting.
    Includes a safety clip to prevent negative floating-point precision noise.
    
    Parameters:
    data: Matrix of shape (n_samples, n_features)
    centroids: Matrix of shape (n_clusters, n_features)
    
    Returns:
    Distance matrix of shape (n_samples, n_clusters)
    """
    # 1. Compute the raw squared distance expansion
    squared_distances = (
        np.sum(data**2, axis=1, keepdims=True) 
        - 2 * np.dot(data, centroids.T) 
        + np.sum(centroids**2, axis=1)
    )
    
    # 2. FIX: Clip any tiny negative values caused by floating-point noise to absolute 0.0
    squared_distances = np.maximum(squared_distances, 0.0)
    
    return np.sqrt(squared_distances)

def apply_winsorization(data: np.ndarray, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> np.ndarray:
    """
    Applies Winsorization thresholds to insulate our evolutionary multi-objective 
    fitness functions from outlier disruptions.
    """
    lower_bound = np.quantile(data, lower_quantile, axis=0)
    upper_bound = np.quantile(data, upper_quantile, axis=0)
    return np.clip(data, lower_bound, upper_bound)
class ClusterIndividual:
    def __init__(self, centroids: np.ndarray):
        """
        Data structure representing a single candidate clustering solution.
        
        Parameters:
        centroids: Array of shape (n_clusters, n_features)
        """
        self.centroids = np.copy(centroids)
        self.fitness = []         # Stores [f1_score, f2_score]
        self.rank = 0             # NSGA-II Pareto dominance rank
        self.crowding_distance = 0.0
        
    def __repr__(self):
        return f"Individual(Rank: {self.rank}, Fitness: {[round(x, 2) for x in self.fitness]})"