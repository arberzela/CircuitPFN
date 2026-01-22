"""
Data generation utilities for synthetic SCM experiments.

This module provides functions to generate synthetic datasets from various
types of Structural Causal Models (SCMs).
"""

import numpy as np
import networkx as nx
from typing import Tuple, Callable, Optional


def generate_random_dag(
    n_nodes: int,
    edge_prob: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random Directed Acyclic Graph (DAG).
    
    Uses Erdős-Rényi model with topological ordering to ensure acyclicity.
    
    Args:
        n_nodes: Number of nodes
        edge_prob: Probability of edge between ordered nodes
        seed: Random seed
        
    Returns:
        Binary adjacency matrix (n_nodes, n_nodes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    adjacency = np.zeros((n_nodes, n_nodes))
    
    # Only add edges from lower to higher index (ensures DAG)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.rand() < edge_prob:
                adjacency[i, j] = 1
    
    return adjacency


def get_topological_order(adjacency: np.ndarray) -> list:
    """
    Get topological ordering of nodes in a DAG.
    
    Args:
        adjacency: Adjacency matrix of DAG
        
    Returns:
        List of node indices in topological order
    """
    G = nx.DiGraph(adjacency)
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXError:
        # If not a DAG, return simple ordering
        return list(range(adjacency.shape[0]))


def generate_linear_gaussian_scm(
    adjacency: np.ndarray,
    n_samples: int,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    noise_std: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data from a Linear-Gaussian SCM.
    
    Each variable is a linear function of its parents plus Gaussian noise:
    X_i = sum_j W_ji * X_j + N(0, noise_std^2)
    
    Args:
        adjacency: Binary adjacency matrix (d, d)
        n_samples: Number of samples to generate
        weight_range: Range for edge weights (min, max)
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Tuple of (data, weighted_adjacency)
        - data: (n_samples, d) array
        - weighted_adjacency: (d, d) array with edge weights
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = adjacency.shape[0]
    topo_order = get_topological_order(adjacency)
    
    # Generate random weights
    weighted_adj = adjacency.copy().astype(float)
    for i in range(d):
        for j in range(d):
            if adjacency[i, j] > 0:
                # Random weight with random sign
                weight = np.random.uniform(weight_range[0], weight_range[1])
                weight *= np.random.choice([-1, 1])
                weighted_adj[i, j] = weight
    
    # Generate data following topological order
    X = np.zeros((n_samples, d))
    
    for node in topo_order:
        # Get parents
        parents = np.where(adjacency[:, node] > 0)[0]
        
        if len(parents) == 0:
            # Root node: just noise
            X[:, node] = np.random.normal(0, noise_std, n_samples)
        else:
            # Linear combination of parents plus noise
            parent_contribution = X[:, parents] @ weighted_adj[parents, node]
            noise = np.random.normal(0, noise_std, n_samples)
            X[:, node] = parent_contribution + noise
    
    return X, weighted_adj


def generate_nonlinear_anm_scm(
    adjacency: np.ndarray,
    n_samples: int,
    noise_std: float = 1.0,
    mechanism: str = 'polynomial',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data from a Non-linear Additive Noise Model (ANM).
    
    Each variable is a non-linear function of its parents plus noise:
    X_i = f_i(PA_i) + N(0, noise_std^2)
    
    Args:
        adjacency: Binary adjacency matrix (d, d)
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise
        mechanism: Type of non-linearity ('polynomial', 'sigmoid', 'mixed')
        seed: Random seed
        
    Returns:
        Tuple of (data, adjacency)
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = adjacency.shape[0]
    topo_order = get_topological_order(adjacency)
    
    # Generate data
    X = np.zeros((n_samples, d))
    
    for node in topo_order:
        parents = np.where(adjacency[:, node] > 0)[0]
        
        if len(parents) == 0:
            # Root node
            X[:, node] = np.random.normal(0, noise_std, n_samples)
        else:
            # Non-linear function of parents
            parent_values = X[:, parents]
            
            if mechanism == 'polynomial':
                # Quadratic: sum(w_i * x_i^2)
                weights = np.random.uniform(0.5, 1.5, len(parents))
                func_output = np.sum(weights * parent_values**2, axis=1)
            
            elif mechanism == 'sigmoid':
                # Sigmoid: sum(w_i * tanh(x_i))
                weights = np.random.uniform(0.5, 2.0, len(parents))
                func_output = np.sum(weights * np.tanh(parent_values), axis=1)
            
            elif mechanism == 'mixed':
                # Random mix of polynomial and sigmoid
                func_output = np.zeros(n_samples)
                for i, p in enumerate(parents):
                    if np.random.rand() > 0.5:
                        # Polynomial
                        func_output += np.random.uniform(0.5, 1.5) * parent_values[:, i]**2
                    else:
                        # Sigmoid
                        func_output += np.random.uniform(0.5, 2.0) * np.tanh(parent_values[:, i])
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")
            
            # Add noise
            noise = np.random.normal(0, noise_std, n_samples)
            X[:, node] = func_output + noise
    
    return X, adjacency


def generate_classification_target(
    X: np.ndarray,
    target_parents: Optional[list] = None,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate binary classification target from features.
    
    Args:
        X: Features (n_samples, d)
        target_parents: Indices of true parent features (None = use all)
        noise_level: Amount of label noise to add
        seed: Random seed
        
    Returns:
        Binary labels (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = X.shape[0]
    
    if target_parents is None:
        target_parents = list(range(X.shape[1]))
    
    # Linear combination of parent features
    X_parents = X[:, target_parents]
    weights = np.random.randn(len(target_parents))
    logits = X_parents @ weights
    
    # Convert to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # Sample labels
    y = (np.random.rand(n_samples) < probs).astype(int)
    
    # Add label noise
    if noise_level > 0:
        flip_mask = np.random.rand(n_samples) < noise_level
        y[flip_mask] = 1 - y[flip_mask]
    
    return y


def generate_synthetic_dataset(
    n_nodes: int = 20,
    n_samples: int = 500,
    edge_prob: float = 0.3,
    scm_type: str = 'linear_gaussian',
    noise_std: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complete synthetic dataset with features, labels, and ground truth graph.
    
    Args:
        n_nodes: Number of features
        n_samples: Number of samples
        edge_prob: Probability of edge in DAG
        scm_type: Type of SCM ('linear_gaussian', 'linear_nongaussian', 'nonlinear_anm')
        noise_std: Noise standard deviation
        seed: Random seed
        
    Returns:
        Tuple of (X, y, true_adjacency)
    """
    # Generate DAG
    adjacency = generate_random_dag(n_nodes, edge_prob, seed)
    
    # Generate features based on SCM type
    if scm_type == 'linear_gaussian':
        X, weighted_adj = generate_linear_gaussian_scm(
            adjacency, n_samples, noise_std=noise_std, seed=seed
        )
        true_adj = adjacency
    
    elif scm_type == 'linear_nongaussian':
        # Still linear but with non-Gaussian noise (Laplace)
        if seed is not None:
            np.random.seed(seed)
        X, weighted_adj = generate_linear_gaussian_scm(
            adjacency, n_samples, noise_std=noise_std, seed=seed
        )
        # Add non-Gaussian component
        X += np.random.laplace(0, noise_std * 0.5, X.shape)
        true_adj = adjacency
    
    elif scm_type == 'nonlinear_anm':
        X, true_adj = generate_nonlinear_anm_scm(
            adjacency, n_samples, noise_std=noise_std, mechanism='mixed', seed=seed
        )
    
    else:
        raise ValueError(f"Unknown SCM type: {scm_type}")
    
    # Generate classification target
    # Use first half of features as true parents
    target_parents = list(range(min(5, n_nodes)))
    y = generate_classification_target(X, target_parents, seed=seed)
    
    return X, y, true_adj
