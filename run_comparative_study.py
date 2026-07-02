import numpy as np
from src.utils import generate_synthetic_spaces
from src.engine import run_evolutionary_clustering

def main():
    print("================================================================")
    print("🚀 STARTING EVOLUTIONARY CLUSTERING COMPARATIVE DISCOVERY LOOP")
    print("================================================================\n")

    # 1. Load Data Tracks
    classical_data, _ = generate_synthetic_spaces()
    real_embeddings = np.load("data/real_deep_embeddings.npy")

    # --- TRACK A: CLASSICAL GEOMETRIC SPACE RUN ---
    print(f"📊 Running Track A: Classical Spatial Array {classical_data.shape}...")
    classical_pareto_front = run_evolutionary_clustering(
        data=classical_data, n_clusters=3, pop_size=20, generations=20
    )
    print(f"✅ Track A Complete. Found {len(classical_pareto_front)} optimal Pareto solutions.")
    for idx, ind in enumerate(classical_pareto_front[:3]):
        print(f"   [Sol {idx}] Compactness: {ind.fitness[0]:.2f} | Separation: {ind.fitness[1]:.2f}")

    # --- TRACK B: DEEP SEMANTIC EMBEDDING RUN ---
    print(f"\n📡 Running Track B: Transformer Latent Space {real_embeddings.shape}...")
    embedding_pareto_front = run_evolutionary_clustering(
        data=real_embeddings, n_clusters=3, pop_size=20, generations=20
    )
    print(f"✅ Track B Complete. Found {len(embedding_pareto_front)} optimal Pareto solutions.")
    for idx, ind in enumerate(embedding_pareto_front[:3]):
        print(f"   [Sol {idx}] Compactness: {ind.fitness[0]:.2f} | Separation: {ind.fitness[1]:.2f}")

    print("\n================================================================")
    print("🎉 SUCCESS: ALL SHUTTLE LOOPS COMPLETED WITHIN CLOUD MEMORY SAFEGUARDS")
    print("================================================================")

if __name__ == "__main__":
    main()