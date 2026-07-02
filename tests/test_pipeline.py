import numpy as np
from src.utils import generate_synthetic_spaces
from src.core_engine import ClusterIndividual, apply_winsorization
from src.objectives import evaluate_compactness, evaluate_separation

print("======= Launching Full Phase 1 Integration Pipeline =======")

# 1. Pull our mock environments
classical_space, embedding_space = generate_synthetic_spaces()
print(f"-> Classical Data Space initialized: {classical_space.shape}")
print(f"-> Deep Embedding Space initialized: {embedding_space.shape}")

# 2. Filter outlier noise using Winsorization thresholds
clean_classical = apply_winsorization(classical_space)

# 3. Simulate creating a random solution candidate (e.g., K=3 clusters)
random_indices = np.random.choice(clean_classical.shape[0], size=3, replace=False)
sample_centroids = clean_classical[random_indices]

candidate_solution = ClusterIndividual(sample_centroids)
print(f"\n-> Chromosome initialized with configuration shape: {candidate_solution.centroids.shape}")

# 4. Run multi-objective fitness calculation sweeps
f1 = evaluate_compactness(clean_classical, candidate_solution.centroids)
f2 = evaluate_separation(candidate_solution.centroids)
candidate_solution.fitness = [f1, f2]

print("\n📈 Calculated Optimization Objectives:")
print(f"   - Compactness Metric (f1 - Minimize): {candidate_solution.fitness[0]:.4f}")
print(f"   - Separation Metric  (f2 - Maximize/Negative Min): {candidate_solution.fitness[1]:.4f}")
print(f"\nResulting Structural Layout: {candidate_solution}")
print("================ Phase 1 Infrastructure Verified ================")
import numpy as np
from src.utils import generate_synthetic_spaces
from src.core_engine import ClusterIndividual
from src.objectives import evaluate_compactness, evaluate_separation
from src.nsga2_ops import fast_non_dominated_sort, assign_crowding_distances

print("======= Launching Phase 2 Multi-Objective Population Audit =======")

# 1. Initialize our standard testing environment 
classical_space, _ = generate_synthetic_spaces()
population = []
population_size = 10  # Mini-population mock run

# 2. Build out a diverse set of random solutions
print(f"Generating a population of {population_size} random clustering configurations...")
for i in range(population_size):
    random_indices = np.random.choice(classical_space.shape[0], size=3, replace=False)
    centroids = classical_space[random_indices]
    
    ind = ClusterIndividual(centroids)
    # Score objectives
    ind.fitness = [
        evaluate_compactness(classical_space, ind.centroids),
        evaluate_separation(ind.centroids)
    ]
    population.append(ind)

# 3. Process the population through our non-dominated sorting layers
sorted_fronts = fast_non_dominated_sort(population)

print("\n📊 Dominance Sorting Layer Results:")
for layer_idx, front in enumerate(sorted_fronts):
    # Assign density variance metrics to the current layer
    assign_crowding_distances(front)
    print(f"  - Front Layer {layer_idx}: Contains {len(front)} individual configurations.")

# 4. Extract and view our top-tier Pareto optimal solution candidates
print("\n🏆 Absolute Pareto-Optimal Front Leaders (Rank 0 Top Choices):")
for elite in sorted_fronts[0]:
    print(f"   -> Compactness: {elite.fitness[0]:.2f} | Separation: {elite.fitness[1]:.2f} | Sparsity: {elite.custom_spacing if hasattr(elite, 'custom_spacing') else elite.crowding_distance}")
    
print("================ Phase 2 Core Engine Operational ================")