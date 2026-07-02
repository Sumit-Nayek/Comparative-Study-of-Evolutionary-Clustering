import numpy as np
from typing import List
from src.core_engine import ClusterIndividual
from src.objectives import evaluate_compactness, evaluate_separation
from src.nsga2_ops import fast_non_dominated_sort, assign_crowding_distances
from src.reproduction import tournament_selection, crossover_centroids, mutate_centroids

def run_evolutionary_clustering(data: np.ndarray, n_clusters: int, pop_size: int = 20, generations: int = 15) -> List[ClusterIndividual]:
    """
    Executes the full NSGA-II multi-objective optimization loop.
    Works seamlessly on classical 2D arrays or deep 384D embedding matrices.
    """
    n_samples, n_features = data.shape
    population = []

    # 1. Initialize Population: Seed individuals with random data samples as centroids
    for _ in range(pop_size):
        random_indices = np.random.choice(n_samples, size=n_clusters, replace=False)
        ind = ClusterIndividual(data[random_indices])
        ind.fitness = [evaluate_compactness(data, ind.centroids), evaluate_separation(ind.centroids)]
        population.append(ind)

    # 2. Main Generational Optimization Loop
    for gen in range(generations):
        # Assign dominance metadata scores to current parent population
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            assign_crowding_distances(front)

        # Generate Offspring Pool (Size N)
        offspring_pool = []
        while len(offspring_pool) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            c1, c2 = crossover_centroids(p1, p2)
            c1 = mutate_centroids(c1, sigma=0.05)
            c2 = mutate_centroids(c2, sigma=0.05)
            
            for child in [c1, c2]:
                child.fitness = [evaluate_compactness(data, child.centroids), evaluate_separation(child.centroids)]
                if len(offspring_pool) < pop_size:
                    offspring_pool.append(child)

        # Merge Parents and Offspring into an elite 2N validation pool
        combined_pool = population + offspring_pool
        sorted_fronts = fast_non_dominated_sort(combined_pool)

        # Environmental Selection Step: Prune 2N pool back down to N
        next_population = []
        for front in sorted_fronts:
            assign_crowding_distances(front)
            
            # Check if the entire front layer fits into the remaining allocation slots
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                # Trimming Step: Sort remaining slots by crowding distance descending (high sparsity wins)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                slots_needed = pop_size - len(next_population)
                next_population.extend(front[:slots_needed])
                break

        population = next_population

    # Return final absolute non-dominated optimization options
    final_fronts = fast_non_dominated_sort(population)
    return final_fronts[0]