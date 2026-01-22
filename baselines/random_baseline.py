"""Random baseline for causal discovery."""

import numpy as np
from typing import Optional


def random_dag(
    n_features: int,
    edge_prob: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random DAG as a baseline.
    
    This serves as a lower bound for causal discovery performance.
    
    Args:
        n_features: Number of features
        edge_prob: Probability of edge between ordered nodes
        seed: Random seed
        
    Returns:
        Binary adjacency matrix (n_features, n_features)
    """
    if seed is not None:
        np.random.seed(seed)
    
    adjacency = np.zeros((n_features, n_features))
    
    # Place edges randomly (only from lower to higher index to ensure DAG)
    for i in range(n_features):
        for j in range(i+1, n_features):
            if np.random.rand() < edge_prob:
                adjacency[i, j] = 1
    
    return adjacency


def run_random_baseline(
    X: np.ndarray,
    edge_prob: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Run random baseline (ignores data, just generates random DAG).
    
    Args:
        X: Data matrix (only used to get n_features)
        edge_prob: Probability of edge
        seed: Random seed
        
    Returns:
        Random binary adjacency matrix
    """
    n_features = X.shape[1]
    return random_dag(n_features, edge_prob, seed)
