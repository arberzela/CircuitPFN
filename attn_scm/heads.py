"""
Structural head identification using entropy-based filtering.

This module implements the methodology for identifying "structural heads" that encode
causal relationships, as opposed to "functional heads" that encode other information.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import entropy


def compute_attention_entropy(attention: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute Shannon entropy of attention distributions.
    
    For each query position, the attention weights form a probability distribution
    over key positions. High entropy indicates diffuse attention (many parents),
    low entropy indicates sparse attention (few parents).
    
    Args:
        attention: Attention weights (batch, heads, d, d) or (heads, d, d)
        normalize: Whether to normalize by log(d) to get values in [0, 1]
        
    Returns:
        Entropy scores per head and query position
    """
    if attention.ndim == 4:
        # Has batch dimension: (batch, heads, d, d)
        batch_size, n_heads, d, _ = attention.shape
        
        # Compute entropy for each batch, head, and query (row)
        entropies = np.zeros((batch_size, n_heads, d))
        
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(d):
                    # Attention distribution for query i
                    attn_dist = attention[b, h, i, :]
                    # Add small epsilon to avoid log(0)
                    attn_dist = attn_dist + 1e-10
                    attn_dist = attn_dist / attn_dist.sum()  # Renormalize
                    entropies[b, h, i] = entropy(attn_dist, base=2)
        
        # Normalize by max possible entropy
        if normalize:
            max_entropy = np.log2(d)
            entropies = entropies / max_entropy
            
    elif attention.ndim == 3:
        # No batch dimension: (heads, d, d)
        n_heads, d, _ = attention.shape
        entropies = np.zeros((n_heads, d))
        
        for h in range(n_heads):
            for i in range(d):
                attn_dist = attention[h, i, :]
                attn_dist = attn_dist + 1e-10
                attn_dist = attn_dist / attn_dist.sum()
                entropies[h, i] = entropy(attn_dist, base=2)
        
        if normalize:
            max_entropy = np.log2(d)
            entropies = entropies / max_entropy
    else:
        raise ValueError(f"Unexpected attention shape: {attention.shape}")
    
    return entropies


def compute_sparsity_score(
    attention: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Compute sparsity score of attention patterns.
    
    Structural heads should have sparse attention (few strong connections).
    
    Args:
        attention: Attention weights (batch, heads, d, d) or (heads, d, d)
        threshold: Threshold for considering a connection as "active"
        
    Returns:
        Sparsity scores per head (lower = sparser)
    """
    if attention.ndim == 4:
        # Average over batch first
        attention = np.mean(attention, axis=0)
    
    n_heads, d, _ = attention.shape
    sparsity_scores = np.zeros(n_heads)
    
    for h in range(n_heads):
        # Count fraction of attention weights above threshold
        active_connections = (attention[h] > threshold).sum()
        total_connections = d * d
        sparsity_scores[h] = active_connections / total_connections
    
    return sparsity_scores


def compute_head_scores(
    attention: np.ndarray,
    method: str = 'entropy'
) -> np.ndarray:
    """
    Compute structural importance scores for each attention head.
    
    Implements Equation 3 from the paper:
    H_score^{(l,h)} = (1/N) * sum_n (1 - H(A_{n,i,:}^{(l,h)}) / log(d))
    
    Lower entropy = higher score = more structural
    
    Args:
        attention: Attention weights (batch, heads, d, d) or (heads, d, d)
        method: Scoring method ('entropy', 'sparsity', 'combined')
        
    Returns:
        Score per head (n_heads,)
    """
    if attention.ndim == 4:
        batch_size, n_heads, d, _ = attention.shape
    elif attention.ndim == 3:
        batch_size = 1
        n_heads, d, _ = attention.shape
    else:
        raise ValueError(f"Unexpected attention shape: {attention.shape}")
    
    if method == 'entropy':
        # Compute normalized entropy
        entropies = compute_attention_entropy(attention, normalize=True)
        
        # Average over batch and query positions
        if entropies.ndim == 3:
            # (batch, heads, d) -> (heads,)
            avg_entropy = np.mean(entropies, axis=(0, 2))
        else:
            # (heads, d) -> (heads,)
            avg_entropy = np.mean(entropies, axis=1)
        
        # Score is inverse of entropy (low entropy = high score)
        scores = 1.0 - avg_entropy
        
    elif method == 'sparsity':
        # Use sparsity score directly
        sparsity = compute_sparsity_score(attention)
        # Invert so that sparse = high score
        scores = 1.0 - sparsity
        
    elif method == 'combined':
        # Combine entropy and sparsity
        entropy_scores = compute_head_scores(attention, method='entropy')
        sparsity_scores = compute_head_scores(attention, method='sparsity')
        scores = 0.5 * entropy_scores + 0.5 * sparsity_scores
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    
    return scores


def identify_structural_heads(
    attention_dict: Dict[int, np.ndarray],
    top_k: int = 5,
    method: str = 'entropy',
    layer_range: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """
    Identify the top-k structural heads across all layers.
    
    Args:
        attention_dict: Dictionary mapping layer_idx -> attention (batch, heads, d, d)
        top_k: Number of top heads to select
        method: Scoring method ('entropy', 'sparsity', 'combined')
        layer_range: Optional tuple (min_layer, max_layer) to restrict search
        
    Returns:
        List of (layer_idx, head_idx) tuples for top-k structural heads
    """
    all_scores = []
    
    for layer_idx, attention in attention_dict.items():
        # Skip layers outside specified range
        if layer_range is not None:
            min_layer, max_layer = layer_range
            if layer_idx < min_layer or layer_idx > max_layer:
                continue
        
        # Compute scores for all heads in this layer
        head_scores = compute_head_scores(attention, method=method)
        
        # Store with layer and head indices
        for head_idx, score in enumerate(head_scores):
            all_scores.append({
                'layer': layer_idx,
                'head': head_idx,
                'score': score
            })
    
    # Sort by score (descending)
    all_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top-k
    top_heads = [(s['layer'], s['head']) for s in all_scores[:top_k]]
    
    return top_heads


def analyze_head_distribution(
    structural_heads: List[Tuple[int, int]],
    n_layers: int = 12
) -> Dict[str, any]:
    """
    Analyze the distribution of structural heads across layers.
    
    This helps validate the "Mid-Layer Hypothesis" from Experiment B.
    
    Args:
        structural_heads: List of (layer_idx, head_idx) tuples
        n_layers: Total number of layers in the model
        
    Returns:
        Dictionary with analysis results
    """
    layer_counts = np.zeros(n_layers)
    
    for layer_idx, _ in structural_heads:
        if layer_idx < n_layers:
            layer_counts[layer_idx] += 1
    
    analysis = {
        'layer_counts': layer_counts,
        'mean_layer': np.mean([l for l, _ in structural_heads]),
        'median_layer': np.median([l for l, _ in structural_heads]),
        'dominant_layers': np.where(layer_counts == layer_counts.max())[0].tolist()
    }
    
    return analysis


def compute_head_stability(
    attention_list: List[np.ndarray],
    head_idx: Tuple[int, int],
    threshold: float = 0.1
) -> float:
    """
    Compute stability of a head's attention pattern across different datasets.
    
    Structural heads should have stable patterns (same parent sets across samples).
    
    Args:
        attention_list: List of attention arrays from different datasets
        head_idx: (layer_idx, head_idx) to analyze
        threshold: Threshold for binarizing attention
        
    Returns:
        Stability score (0 to 1, higher = more stable)
    """
    layer_idx, head_id = head_idx
    
    # Extract attention for this specific head across datasets
    head_patterns = []
    for attention_dict in attention_list:
        if layer_idx in attention_dict:
            # Get attention for this head: (batch, heads, d, d)
            attn = attention_dict[layer_idx]
            if attn.shape[1] > head_id:
                # Average over batch, extract this head
                head_attn = np.mean(attn[:, head_id, :, :], axis=0)
                # Binarize
                binary_pattern = (head_attn > threshold).astype(int)
                head_patterns.append(binary_pattern.flatten())
    
    if len(head_patterns) < 2:
        return 1.0  # Perfect stability by default
    
    # Compute pairwise agreement
    agreements = []
    for i in range(len(head_patterns)):
        for j in range(i+1, len(head_patterns)):
            agreement = (head_patterns[i] == head_patterns[j]).mean()
            agreements.append(agreement)
    
    return np.mean(agreements)
