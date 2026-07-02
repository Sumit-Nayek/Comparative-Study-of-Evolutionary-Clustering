import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from src.core_engine import ClusterIndividual

def plot_pareto_fronts(classical_front: List[ClusterIndividual], embedding_front: List[ClusterIndividual]):
    """
    Generates high-resolution publication-ready scatter plots of the discovered 
    Pareto optimal fronts for both testing spaces.
    """
    # Set style properties for academic formatting
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Extract Classical Space Metrics
    class_f1 = [ind.fitness[0] for ind in classical_front]
    # Multiply by -1 to flip the negative minimization trick back to a true positive Maximization axis
    class_f2 = [-ind.fitness[1] for ind in classical_front]
    
    ax1.scatter(class_f1, class_f2, color='#1f77b4', s=60, edgecolors='black', alpha=0.8, label='Pareto Optimal Set')
    ax1.set_title("Track A: Classical Spatial Array (2D)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Compactness Variance (f1 - Lower is Better)", fontsize=10)
    ax1.set_ylabel("Inter-Cluster Separation (f2 - Higher is Better)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # 2. Extract Latent Embedding Space Metrics
    embed_f1 = [ind.fitness[0] for ind in embedding_front]
    embed_f2 = [-ind.fitness[1] for ind in embedding_front]
    
    ax2.scatter(embed_f1, embed_f2, color='#ff7f0e', s=60, edgecolors='black', alpha=0.8, label='Pareto Optimal Set')
    ax2.set_title("Track B: Transformer Latent Space (384D)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Compactness Variance (f1 - Lower is Better)", fontsize=10)
    ax2.set_ylabel("Inter-Cluster Separation (f2 - Higher is Better)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the output directly to the workspace for the paper draft
    output_path = "data/pareto_comparison_fronts.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Graphical Chart successfully exported and saved to: '{output_path}'")