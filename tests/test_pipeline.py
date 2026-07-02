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