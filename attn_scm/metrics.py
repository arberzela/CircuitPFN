"""
Evaluation metrics for causal graph comparison.

This module implements metrics for comparing predicted and ground-truth causal graphs,
including Structural Hamming Distance (SHD) and F1 score.
"""

import numpy as np
from typing import Tuple, Dict
import networkx as nx


def structural_hamming_distance(
    pred: np.ndarray,
    true: np.ndarray,
    double_for_anticausal: bool = True
) -> int:
    """
    Compute Structural Hamming Distance (SHD) between two graphs.
    
    SHD counts the number of edge additions, deletions, and reversals needed
    to transform the predicted graph into the true graph.
    
    Args:
        pred: Predicted adjacency matrix (d, d), binary
        true: True adjacency matrix (d, d), binary
        double_for_anticausal: Whether to count reversed edges as 2 errors
        
    Returns:
        SHD value (lower is better, 0 is perfect)
    """
    # Binarize if needed
    pred_bin = (pred > 0).astype(int)
    true_bin = (true > 0).astype(int)
    
    d = pred_bin.shape[0]
    errors = 0
    
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            
            pred_edge = pred_bin[i, j]
            true_edge = true_bin[i, j]
            
            if pred_edge != true_edge:
                # Edge mismatch
                if pred_edge == 1 and true_edge == 0:
                    # False positive (extra edge)
                    # Check if this is a reversal (edge exists in opposite direction)
                    if double_for_anticausal and true_bin[j, i] == 1 and pred_bin[j, i] == 0:
                        # This is a reversal, count as 2
                        errors += 2
                    else:
                        # Just an extra edge
                        errors += 1
                elif pred_edge == 0 and true_edge == 1:
                    # False negative (missing edge)
                    # Check if this is part of a reversal (already counted)
                    if double_for_anticausal and pred_bin[j, i] == 1 and true_bin[j, i] == 0:
                        # Already counted as reversal
                        pass
                    else:
                        errors += 1
    
    # Simpler alternative: just count differences
    # This avoids double-counting issues
    if not double_for_anticausal:
        errors = np.sum(pred_bin != true_bin)
    
    return int(errors)


def compute_precision_recall_f1(
    pred: np.ndarray,
    true: np.ndarray,
    ignore_direction: bool = False
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for edge prediction.
    
    Args:
        pred: Predicted adjacency matrix (d, d)
        true: True adjacency matrix (d, d)
        ignore_direction: If True, treat graph as undirected
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    # Binarize
    pred_bin = (pred > 0).astype(int)
    true_bin = (true > 0).astype(int)
    
    if ignore_direction:
        # Convert to undirected by symmetrizing
        pred_bin = np.maximum(pred_bin, pred_bin.T)
        true_bin = np.maximum(true_bin, true_bin.T)
    
    # Remove diagonal
    np.fill_diagonal(pred_bin, 0)
    np.fill_diagonal(true_bin, 0)
    
    # Flatten
    pred_flat = pred_bin.flatten()
    true_flat = true_bin.flatten()
    
    # Compute metrics
    true_positives = np.sum((pred_flat == 1) & (true_flat == 1))
    false_positives = np.sum((pred_flat == 1) & (true_flat == 0))
    false_negatives = np.sum((pred_flat == 0) & (true_flat == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return precision, recall, f1


def compute_graph_metrics(
    pred: np.ndarray,
    true: np.ndarray
) -> Dict[str, float]:
    """
    Compute a comprehensive set of graph comparison metrics.
    
    Args:
        pred: Predicted adjacency matrix (d, d)
        true: True adjacency matrix (d, d)
        
    Returns:
        Dictionary of metrics
    """
    # SHD
    shd = structural_hamming_distance(pred, true)
    
    # Precision, Recall, F1 (directed)
    prec_dir, rec_dir, f1_dir = compute_precision_recall_f1(pred, true, ignore_direction=False)
    
    # Precision, Recall, F1 (undirected)
    prec_undir, rec_undir, f1_undir = compute_precision_recall_f1(pred, true, ignore_direction=True)
    
    # Edge counts
    pred_edges = int((pred > 0).sum())
    true_edges = int((true > 0).sum())
    
    metrics = {
        'shd': shd,
        'precision_directed': prec_dir,
        'recall_directed': rec_dir,
        'f1_directed': f1_dir,
        'precision_undirected': prec_undir,
        'recall_undirected': rec_undir,
        'f1_undirected': f1_undir,
        'pred_num_edges': pred_edges,
        'true_num_edges': true_edges,
        'edge_diff': abs(pred_edges - true_edges)
    }
    
    return metrics


def normalized_shd(
    pred: np.ndarray,
    true: np.ndarray
) -> float:
    """
    Compute normalized SHD (SHD divided by maximum possible edges).
    
    Args:
        pred: Predicted adjacency matrix (d, d)
        true: True adjacency matrix (d, d)
        
    Returns:
        Normalized SHD in [0, 1] (lower is better)
    """
    shd = structural_hamming_distance(pred, true)
    d = pred.shape[0]
    max_edges = d * (d - 1)  # Maximum number of directed edges
    
    return shd / max_edges if max_edges > 0 else 0.0
