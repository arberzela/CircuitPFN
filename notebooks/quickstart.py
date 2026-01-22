"""Example: Quick start with Attn-SCM."""

import numpy as np
from attn_scm.core import AttnSCM
from utils import generate_synthetic_dataset, plot_comparison

# Generate synthetic data
print("Generating synthetic causal dataset...")
X, y, true_adjacency = generate_synthetic_dataset(
    n_nodes=15,
    n_samples=300,
    edge_prob=0.25,
    scm_type='linear_gaussian',
    seed=42
)

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"True graph has {(true_adjacency > 0).sum()} edges")

# Initialize Attn-SCM
print("\nInitializing Attn-SCM...")
model = AttnSCM(
    top_k_heads=5,              # Number of structural attention heads
    threshold_method='otsu',     # Automatic threshold selection
    directionality_method='asymmetry',
    device='cpu'
)

# Extract causal graph
print("Extracting causal graph from TabPFN attention...")
predicted_adjacency = model.fit(X, y)

print(f"Predicted graph has {(predicted_adjacency > 0).sum()} edges")

# Get Markov Blanket of first feature
mb = model.get_markov_blanket(0)
print(f"\nMarkov Blanket of feature 0: {mb}")

# Visualize comparison
print("\nGenerating comparison plot...")
plot_comparison(
    predicted_adjacency,
    true_adjacency,
    save_path='example_comparison.png'
)

print("\nDone! Saved comparison to example_comparison.png")
