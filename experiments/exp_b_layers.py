#!/usr/bin/env python3
"""
Experiment B: Layer-Wise Localization Test

Tests the "Mid-Layer Hypothesis" by extracting graphs from each layer independently.

Expected: U-shaped error curve with minimum at mid-layers (4-7)
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from attn_scm.core import AttnSCM
from attn_scm.metrics import structural_hamming_distance, compute_graph_metrics
from utils import generate_synthetic_dataset, save_results, plot_layer_analysis


def run_experiment_b(
    num_datasets: int = 50,
    n_samples: int = 500,
    n_features: int = 20,
    n_layers: int = 12,
    output_dir: str = "results/exp_b",
    seed: int = 42
):
    """
    Run Experiment B: Layer-wise Localization Test.
    
    Args:
        num_datasets: Number of synthetic datasets
        n_samples: Samples per dataset
        n_features: Features per dataset
        n_layers: Number of layers in TabPFN (default 12)
        output_dir: Output directory
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXPERIMENT B: Layer-Wise Localization Test")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of datasets: {num_datasets}")
    print(f"  - Samples per dataset: {n_samples}")
    print(f"  - Features per dataset: {n_features}")
    print(f"  - Number of layers: {n_layers}")
    print("="*80)
    
    # Results storage
    all_results = []
    
    # Initialize model
    model = AttnSCM(
        top_k_heads=5,
        threshold_method='otsu',
        device='cpu'
    )
    
    # Test each layer
    for layer_idx in range(n_layers):
        print(f"\nTesting Layer {layer_idx}")
        print("-"*80)
        
        layer_shd_scores = []
        layer_f1_scores = []
        
        for dataset_idx in tqdm(range(num_datasets), desc=f"Layer {layer_idx}"):
            # Generate dataset
            X, y, true_adj = generate_synthetic_dataset(
                n_nodes=n_features,
                n_samples=n_samples,
                edge_prob=0.3,
                scm_type='linear_gaussian',
                seed=seed + dataset_idx
            )
            
            try:
                # Extract graph from this specific layer only
                pred_adj = model.extract_for_layer(
                    X, y,
                    layer_idx=layer_idx,
                    head_indices=None  # Use all heads in this layer
                )
                
                # Compute metrics
                metrics = compute_graph_metrics(pred_adj, true_adj)
                shd = metrics['shd']
                f1 = metrics['f1_directed']
                
                layer_shd_scores.append(shd)
                layer_f1_scores.append(f1)
                
                # Store results
                all_results.append({
                    'layer': layer_idx,
                    'dataset_idx': dataset_idx,
                    'shd': shd,
                    'f1_directed': f1,
                    **metrics
                })
            
            except Exception as e:
                print(f"\nError with layer {layer_idx}, dataset {dataset_idx}: {e}")
                all_results.append({
                    'layer': layer_idx,
                    'dataset_idx': dataset_idx,
                    'error': str(e)
                })
        
        # Print layer summary
        if layer_shd_scores:
            mean_shd = np.mean(layer_shd_scores)
            mean_f1 = np.mean(layer_f1_scores)
            print(f"Layer {layer_idx} - Mean SHD: {mean_shd:.2f}, Mean F1: {mean_f1:.3f}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save raw results
    csv_path = output_path / "raw_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to: {csv_path}")
    
    # Compute layer-wise summary
    layer_summary = df_results.groupby('layer').agg({
        'shd': ['mean', 'std', 'median'],
        'f1_directed': ['mean', 'std', 'median']
    }).round(3)
    
    print("\n" + "="*80)
    print("LAYER-WISE SUMMARY")
    print("="*80)
    print(layer_summary)
    
    # Save summary
    summary_path = output_path / "layer_summary.csv"
    layer_summary.to_csv(summary_path)
    
    # Identify best layer
    layer_mean_shd = df_results.groupby('layer')['shd'].mean()
    best_layer = layer_mean_shd.idxmin()
    best_shd = layer_mean_shd[best_layer]
    
    print(f"\nBest Layer: {best_layer} (Mean SHD: {best_shd:.2f})")
    
    # Visualize layer-wise performance
    print("\nGenerating visualizations...")
    
    # Plot SHD vs Layer
    plot_layer_analysis(
        layer_mean_shd.to_dict(),
        save_path=output_path / "layer_shd_curve.png"
    )
    
    # Additional plot: F1 vs Layer
    layer_mean_f1 = df_results.groupby('layer')['f1_directed'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        layer_mean_f1.index,
        layer_mean_f1.values,
        marker='o',
        linewidth=2,
        markersize=8,
        color='darkgreen'
    )
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise F1 Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(best_layer, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path / "layer_f1_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("EXPERIMENT B COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print(f"Best layer for causal extraction: Layer {best_layer}")
    
    return df_results, layer_summary, best_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment B: Layer-wise Localization')
    parser.add_argument('--num_datasets', type=int, default=50, help='Number of datasets')
    parser.add_argument('--n_samples', type=int, default=500, help='Samples per dataset')
    parser.add_argument('--n_features', type=int, default=20, help='Features per dataset')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--output_dir', type=str, default='results/exp_b', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_experiment_b(
        num_datasets=args.num_datasets,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_layers=args.n_layers,
        output_dir=args.output_dir,
        seed=args.seed
    )
