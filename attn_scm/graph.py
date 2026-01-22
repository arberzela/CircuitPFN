"""
Graph construction and post-processing utilities.

This module implements adjacency matrix aggregation, thresholding, and
directionality enforcement as described in the paper.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import precision_recall_fscore_support


def aggregate_heads_to_adjacency(
    attention_dict: Dict[int, np.ndarray],
    structural_heads: List[Tuple[int, int]],
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Aggregate selected structural heads into a raw adjacency matrix.

    Implements Equation 4 from the paper:
    W_raw[i,j] = (1/|S_causal|) * sum_{(l,h) in S_causal} E_n[A_n^{(l,h)}[i,j]]

    Args:
        attention_dict: Dictionary mapping layer_idx -> attention (batch, heads, d, d)
        structural_heads: List of (layer_idx, head_idx) tuples
        aggregation: How to combine heads ('mean', 'max', 'sum')

    Returns:
        Raw adjacency matrix (d, d)
    """
    head_matrices = []

    for layer_idx, head_idx in structural_heads:
        if layer_idx not in attention_dict:
            continue

        # Get attention for this layer: (batch, heads, d, d)
        layer_attention = attention_dict[layer_idx]

        print(f"DEBUG: Processing layer {layer_idx}, head {head_idx}, attention shape: {layer_attention.shape}")

        # Shape checking
        if layer_attention.ndim == 4:
            batch_size, n_heads, d, d_k = layer_attention.shape
            if head_idx >= n_heads:
                continue
            # Extract this head and average over batch
            head_attn = np.mean(layer_attention[:, head_idx, :, :], axis=0)
            print(f"DEBUG: Extracted head attention shape (4D case): {head_attn.shape}")
        elif layer_attention.ndim == 3:
            # Already batch-averaged: (heads, d, d)
            n_heads, d, d_k = layer_attention.shape
            if head_idx >= n_heads:
                continue
            head_attn = layer_attention[head_idx, :, :]
            print(f"DEBUG: Extracted head attention shape (3D case): {head_attn.shape}")
        else:
            print(f"DEBUG: Unexpected ndim: {layer_attention.ndim}")
            continue

        head_matrices.append(head_attn)
    
    if not head_matrices:
        raise ValueError("No valid attention matrices found for structural heads")

    # Validate all matrices have the same shape and are square
    first_shape = head_matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError(f"Expected square 2D matrices, but got shape {first_shape}")

    for i, mat in enumerate(head_matrices):
        if mat.shape != first_shape:
            raise ValueError(f"Head matrix {i} has shape {mat.shape}, expected {first_shape}")

    # Stack all head matrices
    stacked = np.stack(head_matrices, axis=0)  # (n_heads, d, d)
    print(f"DEBUG: Stacked attention shape: {stacked.shape}")
    
    # Aggregate across heads
    if aggregation == 'mean':
        adj_raw = np.nanmean(stacked, axis=0)
    elif aggregation == 'max':
        adj_raw = np.nanmax(stacked, axis=0)
    elif aggregation == 'sum':
        adj_raw = np.nansum(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Clean up any remaining NaNs or infinities
    adj_raw = np.nan_to_num(adj_raw, nan=0.0, posinf=0.0, neginf=0.0)

    return adj_raw


def threshold_adjacency(
    adjacency: np.ndarray,
    method: str = 'otsu',
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Apply thresholding to enforce sparsity in the adjacency matrix.

    Args:
        adjacency: Raw adjacency matrix (d, d)
        method: Thresholding method ('otsu', 'fixed', 'top_k', 'percentile')
        threshold: Threshold value (required for 'fixed' method)

    Returns:
        Thresholded adjacency matrix (d, d)
    """
    # Handle NaNs and infinities
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == 'fixed':
        if threshold is None:
            raise ValueError("threshold must be provided for 'fixed' method")
        return np.where(adjacency > threshold, adjacency, 0.0)
    
    elif method == 'otsu':
        # Otsu's method for automatic thresholding
        thresh = otsu_threshold(adjacency.flatten())
        return np.where(adjacency > thresh, adjacency, 0.0)
    
    elif method == 'top_k':
        # Keep top-k edges per node
        k = threshold if threshold is not None else 2
        k = int(k)
        result = np.zeros_like(adjacency)
        d = adjacency.shape[0]
        for i in range(d):
            # Get top-k incoming edges for node i
            top_k_idx = np.argsort(adjacency[i, :])[-k:]
            result[i, top_k_idx] = adjacency[i, top_k_idx]
        return result
    
    elif method == 'percentile':
        # Keep edges above a certain percentile
        pct = threshold if threshold is not None else 90
        thresh = np.percentile(adjacency[adjacency > 0], pct)
        return np.where(adjacency > thresh, adjacency, 0.0)
    
    else:
        raise ValueError(f"Unknown thresholding method: {method}")


def otsu_threshold(values: np.ndarray) -> float:
    """
    Compute Otsu's threshold for automatic binarization.
    
    Args:
        values: Flattened array of values
        
    Returns:
        Optimal threshold value
    """
    # Remove zeros and negative values
    values = values[values > 0]
    
    if len(values) == 0:
        return 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(float)
    hist /= hist.sum()
    
    # Compute cumulative sums
    weight_bg = np.cumsum(hist)
    weight_fg = 1.0 - weight_bg
    
    # Compute cumulative means
    mean_bg = np.cumsum(hist * bin_centers) / (weight_bg + 1e-10)
    
    total_mean = np.sum(hist * bin_centers)
    mean_fg = (total_mean - np.cumsum(hist * bin_centers)) / (weight_fg + 1e-10)
    
    # Compute between-class variance
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    
    # Find threshold that maximizes variance
    optimal_idx = np.argmax(variance)
    optimal_threshold = bin_centers[optimal_idx]
    
    return optimal_threshold


def enforce_directionality(
    adjacency: np.ndarray,
    method: str = 'asymmetry'
) -> np.ndarray:
    """
    Enforce directed edges by breaking symmetry.
    
    Implements Equation 5 from the paper:
    if W_ij < W_ji => W_ij := 0
    
    The intuition is that the stronger direction represents the true causal edge.
    
    Args:
        adjacency: Adjacency matrix (d, d), possibly bidirectional
        method: Method for enforcing direction ('asymmetry', 'dag')
        
    Returns:
        Directed adjacency matrix (d, d)
    """
    if method == 'asymmetry':
        # Keep the stronger of each bidirectional edge
        result = adjacency.copy()
        d = adjacency.shape[0]
        
        for i in range(d):
            for j in range(i+1, d):  # Upper triangle only
                if adjacency[i, j] > 0 and adjacency[j, i] > 0:
                    # Bidirectional edge exists
                    if adjacency[i, j] > adjacency[j, i]:
                        # i -> j is stronger
                        result[j, i] = 0.0
                    else:
                        # j -> i is stronger
                        result[i, j] = 0.0
        
        return result
    
    elif method == 'dag':
        # Enforce DAG constraint by removing cycles
        result = adjacency.copy()
        
        # Convert to binary graph
        binary = (result > 0).astype(int)
        
        # Remove edges to make it a DAG
        result = make_dag(result, binary)
        
        return result
    
    else:
        raise ValueError(f"Unknown directionality method: {method}")


def make_dag(adjacency: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """
    Remove edges to convert graph to a DAG.

    Uses a greedy approach: iteratively remove the weakest edge in cycles.

    Args:
        adjacency: Weighted adjacency matrix
        binary: Binary adjacency matrix

    Returns:
        DAG adjacency matrix
    """
    result = adjacency.copy()

    # Clean binary matrix before creating graph
    binary_clean = np.nan_to_num(binary, nan=0.0, posinf=0.0, neginf=0.0).astype(int)

    # Create graph manually to avoid NetworkX interpretation issues
    d = binary_clean.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if binary_clean[i, j] > 0:
                G.add_edge(i, j)
    
    # While there are cycles
    max_iterations = 1000
    iteration = 0
    while not nx.is_directed_acyclic_graph(G) and iteration < max_iterations:
        try:
            # Find a cycle
            cycle = nx.find_cycle(G)
            
            # Find weakest edge in cycle
            weakest_edge = None
            min_weight = float('inf')
            
            for u, v in cycle:
                weight = result[u, v]
                if weight < min_weight:
                    min_weight = weight
                    weakest_edge = (u, v)
            
            # Remove weakest edge
            if weakest_edge:
                u, v = weakest_edge
                result[u, v] = 0.0
                G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
        
        iteration += 1
    
    return result


def extract_markov_blanket(
    adjacency: np.ndarray,
    target_idx: int
) -> List[int]:
    """
    Extract the Markov Blanket of a target variable from the adjacency matrix.
    
    Markov Blanket consists of:
    - Parents of target
    - Children of target
    - Spouses (other parents of children)
    
    Args:
        adjacency: Binary or weighted adjacency matrix (d, d)
        target_idx: Index of target variable
        
    Returns:
        List of feature indices in the Markov Blanket
    """
    d = adjacency.shape[0]
    
    # Binarize adjacency
    binary_adj = (adjacency > 0).astype(int)
    
    # Find parents (incoming edges)
    parents = set(np.where(binary_adj[:, target_idx] > 0)[0])
    
    # Find children (outgoing edges)
    children = set(np.where(binary_adj[target_idx, :] > 0)[0])
    
    # Find spouses (other parents of children)
    spouses = set()
    for child in children:
        child_parents = set(np.where(binary_adj[:, child] > 0)[0])
        spouses.update(child_parents - {target_idx})
    
    # Combine all
    markov_blanket = parents | children | spouses
    
    # Remove target itself
    markov_blanket.discard(target_idx)
    
    return sorted(list(markov_blanket))


def validate_dag(adjacency: np.ndarray) -> Dict[str, any]:
    """
    Validate that the adjacency matrix represents a valid DAG.

    Args:
        adjacency: Adjacency matrix (d, d)

    Returns:
        Dictionary with validation results
    """
    # Clean the adjacency matrix first
    adjacency_clean = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    binary = (adjacency_clean > 0).astype(int)

    # Ensure it's a proper square matrix
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError(f"Adjacency matrix must be square 2D, got shape {binary.shape}")

    # Use explicit converter to avoid ambiguity
    try:
        if hasattr(nx, 'from_numpy_array'):
            G = nx.from_numpy_array(binary, create_using=nx.DiGraph)
        else:
            G = nx.from_numpy_matrix(binary, create_using=nx.DiGraph)
    except Exception as e:
        # Fallback: create graph manually from binary matrix
        d = binary.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(d))
        for i in range(d):
            for j in range(d):
                if binary[i, j] > 0:
                    G.add_edge(i, j)
    
    is_dag = nx.is_directed_acyclic_graph(G)
    
    validation = {
        'is_dag': is_dag,
        'num_edges': int(binary.sum()),
        'num_nodes': adjacency.shape[0],
        'sparsity': 1.0 - (binary.sum() / (adjacency.shape[0] ** 2)),
        'has_cycles': not is_dag
    }
    
    if not is_dag:
        try:
            cycle = nx.find_cycle(G)
            validation['example_cycle'] = cycle
        except:
            validation['example_cycle'] = None
    
    return validation
