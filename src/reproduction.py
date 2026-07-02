import numpy as np
from typing import List
from src.core_engine import ClusterIndividual

def crossover_centroids(parent1: ClusterIndividual, parent2: ClusterIndividual, crossover_rate: float = 0.8) -> tuple:
    """
    Applies vectorized arithmetic blending between two parent centroid configurations.
    """
    if np.random.rand() > crossover_rate:
        return ClusterIndividual(parent1.centroids), ClusterIndividual(parent2.centroids)
        
    # Generate a random blending weight vector matching the shape of the centroids
    alpha = np.random.rand(*parent1.centroids.shape)
    
    child1_centroids = alpha * parent1.centroids + (1.0 - alpha) * parent2.centroids
    child2_centroids = (1.0 - alpha) * parent1.centroids + alpha * parent2.centroids
    
    return ClusterIndividual(child1_centroids), ClusterIndividual(child2_centroids)

def mutate_centroids(individual: ClusterIndividual, mutation_rate: float = 0.2, sigma: float = 0.1) -> ClusterIndividual:
    """
    Applies Gaussian noise to random rows (centroids) within an individual to encourage exploration.
    """
    centroids = np.copy(individual.centroids)
    n_clusters, n_features = centroids.shape
    
    for i in range(n_clusters):
        if np.random.rand() < mutation_rate:
            # Generate local spatial mutation noise
            noise = np.random.normal(0, sigma, size=n_features)
            centroids[i] += noise
            
    return ClusterIndividual(centroids)
def tournament_selection(population: List[ClusterIndividual]) -> ClusterIndividual:
    """
    Selects a high-quality parent solution using Non-Dominated rank and Crowding Distance attributes.
    """
    # Grab two candidate solutions completely at random
    idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
    ind1, ind2 = population[idx1], population[idx2]
    
    # Selection rule 1: Prioritize lower/better Pareto rank front layer
    if ind1.rank < ind2.rank:
        return ind1
    elif ind2.rank < ind1.rank:
        return ind2
        
    # Selection rule 2: If ranks tie, choose the individual with less cluster density/higher sparsity
    if ind1.crowding_distance > ind2.crowding_distance:
        return ind1
    return ind2