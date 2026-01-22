"""Visualization utilities for causal graphs and experimental results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional, List, Tuple


def plot_adjacency_matrix(
    adjacency: np.ndarray,
    title: str = "Adjacency Matrix",
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot adjacency matrix as a heatmap.
    
    Args:
        adjacency: Adjacency matrix (d, d)
        title: Plot title
        feature_names: Names for features
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Binarize for visualization
    binary_adj = (adjacency > 0).astype(int)
    
    sns.heatmap(
        binary_adj,
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=feature_names or range(adjacency.shape[1]),
        yticklabels=feature_names or range(adjacency.shape[0]),
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('To (Child)', fontsize=12)
    ax.set_ylabel('From (Parent)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_graph_network(
    adjacency: np.ndarray,
    title: str = "Causal Graph",
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    node_size: int = 1500,
    highlight_nodes: Optional[List[int]] = None
):
    """
    Plot causal graph as a network diagram.
    
    Args:
        adjacency: Adjacency matrix (d, d)
        title: Plot title
        feature_names: Names for features
        save_path: Path to save figure
        figsize: Figure size
        node_size: Size of nodes
        highlight_nodes: Node indices to highlight
    """
    binary_adj = (adjacency > 0).astype(int)
    G = nx.DiGraph(binary_adj)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use hierarchical layout for DAGs
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Node colors
    node_colors = ['lightblue'] * len(G.nodes())
    if highlight_nodes:
        for node in highlight_nodes:
            if node < len(node_colors):
                node_colors[node] = 'orange'
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Labels
    labels = {}
    for i in range(len(G.nodes())):
        if feature_names and i < len(feature_names):
            labels[i] = feature_names[i]
        else:
            labels[i] = f"X{i}"
    
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=10,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison(
    pred_adj: np.ndarray,
    true_adj: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot side-by-side comparison of predicted and true graphs.
    
    Args:
        pred_adj: Predicted adjacency matrix
        true_adj: True adjacency matrix
        feature_names: Feature names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # True graph
    sns.heatmap(
        (true_adj > 0).astype(int),
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0.5,
        xticklabels=feature_names or range(true_adj.shape[1]),
        yticklabels=feature_names or range(true_adj.shape[0]),
        ax=axes[0]
    )
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('To', fontsize=12)
    axes[0].set_ylabel('From', fontsize=12)
    
    # Predicted graph
    sns.heatmap(
        (pred_adj > 0).astype(int),
        cmap='Oranges',
        cbar=True,
        square=True,
        linewidths=0.5,
        xticklabels=feature_names or range(pred_adj.shape[1]),
        yticklabels=feature_names or range(pred_adj.shape[0]),
        ax=axes[1]
    )
    axes[1].set_title('Predicted (Attn-SCM)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('To', fontsize=12)
    axes[1].set_ylabel('From', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_layer_analysis(
    layer_scores: dict,
    save_path: Optional[str] = None
):
    """
    Plot layer-wise analysis for Experiment B.
    
    Args:
        layer_scores: Dictionary mapping layer_idx -> SHD or F1
        save_path: Path to save figure
    """
    layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(layers, scores, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('SHD (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Causal Graph Extraction Quality', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    # Highlight best layer
    best_layer = min(layer_scores, key=layer_scores.get)
    best_score = layer_scores[best_layer]
    ax.axvline(best_layer, color='red', linestyle='--', alpha=0.5, label=f'Best Layer: {best_layer}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    results_dict: dict,
    metric_name: str = 'SHD',
    save_path: Optional[str] = None
):
    """
    Plot comparison of metrics across different methods.
    
    Args:
        results_dict: Dictionary mapping method_name -> list of scores
        metric_name: Name of metric to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results_dict.keys())
    scores = list(results_dict.values())
    
    bp = ax.boxplot(scores, labels=methods, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
