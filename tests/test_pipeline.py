import numpy as np
from src.core_engine import calculate_distance_matrix, apply_winsorization

# 1. Create a tiny test array (3 data samples, 2 dimensions)
mock_data = np.array([[1.0, 2.0], [10.0, 11.0], [1.5, 2.5]])
# 2. Set up 2 dummy cluster centers
mock_centroids = np.array([[1.1, 2.1], [9.5, 10.5]])

print("--- Launching Codespaces Diagnostics ---")
clean_data = apply_winsorization(mock_data)
dist_matrix = calculate_distance_matrix(clean_data, mock_centroids)

print("Calculated Distance Matrix shape:", dist_matrix.shape)
print("Distance Values:\n", dist_matrix)
print("--- Phase 1 Base Infrastructure Functional ---")