#!/usr/bin/env python3
"""
Experiment A: Synthetic Benchmark Recovery

Tests topological accuracy of Attn-SCM on 100 synthetic SCMs with varying mechanisms.

Metrics: Structural Hamming Distance (SHD), F1-Score
Baselines: PC Algorithm, NOTEARS, Random
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from attn_scm.core import AttnSCM
from attn_scm.metrics import compute_graph_metrics, structural_hamming_distance
from baselines import run_pc_algorithm, run_notears, run_random_baseline
from utils import generate_synthetic_dataset, save_results, plot_metrics_comparison


def run_experiment_a(
    num_datasets: int = 100,
    n_samples: int = 500,
    n_features: int = 20,
    edge_prob: float = 0.3,
    scm_types: list = None,
    output_dir: str = "results/exp_a",
    seed: int = 42
):
    """
    Run Experiment A: Synthetic Benchmark Recovery.
    
    Args:
        num_datasets: Number of synthetic datasets to generate
        n_samples: Number of samples per dataset
        n_features: Number of features per dataset
        edge_prob: Probability of edge in ground-truth DAG
        scm_types: List of SCM types to test
        output_dir: Directory to save results
        seed: Random seed
    """
    if scm_types is None:
        scm_types = ['linear_gaussian', 'linear_nongaussian', 'nonlinear_anm']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXPERIMENT A: Synthetic Benchmark Recovery")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of datasets: {num_datasets}")
    print(f"  - Samples per dataset: {n_samples}")
    print(f"  - Features per dataset: {n_features}")
    print(f"  - SCM types: {scm_types}")
    print("="*80)
    
    # Results storage
    all_results = []
    
    # Initialize methods
    methods = {
        'AttnSCM': lambda X, y: AttnSCM(top_k_heads=5, threshold_method='otsu'),
        'PC': lambda X, y: None,  # PC doesn't need instantiation
        'NOTEARS': lambda X, y: None,
        'Random': lambda X, y: None
    }
    
    # Run experiments
    for scm_type in scm_types:
        print(f"\nTesting SCM type: {scm_type}")
        print("-"*80)
        
        for dataset_idx in tqdm(range(num_datasets), desc=f"{scm_type}"):
            # Generate synthetic dataset
            X, y, true_adj = generate_synthetic_dataset(
                n_nodes=n_features,
                n_samples=n_samples,
                edge_prob=edge_prob,
                scm_type=scm_type,
                seed=seed + dataset_idx
            )
            
            # Test each method
            for method_name in methods:
                try:
                    # Run method
                    if method_name == 'AttnSCM':
                        model = AttnSCM(
                            top_k_heads=5,
                            threshold_method='otsu',
                            device='cpu'
                        )
                        pred_adj = model.fit(X, y)
                    
                    elif method_name == 'PC':
                        pred_adj = run_pc_algorithm(X, alpha=0.05)
                    
                    elif method_name == 'NOTEARS':
                        pred_adj = run_notears(X, lambda1=0.1, w_threshold=0.3)
                    
                    elif method_name == 'Random':
                        pred_adj = run_random_baseline(X, edge_prob=edge_prob, seed=seed+dataset_idx)
                    
                    # Compute metrics
                    metrics = compute_graph_metrics(pred_adj, true_adj)
                    
                    # Store results
                    result = {
                        'scm_type': scm_type,
                        'dataset_idx': dataset_idx,
                        'method': method_name,
                        **metrics
                    }
                    all_results.append(result)
                
                except Exception as e:
                    print(f"\nError with {method_name} on dataset {dataset_idx}: {e}")
                    # Store failed result
                    all_results.append({
                        'scm_type': scm_type,
                        'dataset_idx': dataset_idx,
                        'method': method_name,
                        'error': str(e)
                    })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save raw results
    csv_path = output_path / "raw_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to: {csv_path}")
    
    # Compute summary statistics
    summary = df_results.groupby(['scm_type', 'method']).agg({
        'shd': ['mean', 'std', 'median'],
        'f1_directed': ['mean', 'std', 'median'],
        'f1_undirected': ['mean', 'std', 'median']
    }).round(3)
    
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(summary)
    
    # Save summary
    summary_path = output_path / "summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary to: {summary_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # SHD comparison
    for scm_type in scm_types:
        df_scm = df_results[df_results['scm_type'] == scm_type]
        
        if 'shd' in df_scm.columns:
            shd_by_method = {
                method: df_scm[df_scm['method'] == method]['shd'].dropna().values
                for method in df_scm['method'].unique()
            }
            
            plot_metrics_comparison(
                shd_by_method,
                metric_name='SHD',
                save_path=output_path / f"shd_comparison_{scm_type}.png"
            )
    
    # F1 comparison
    for scm_type in scm_types:
        df_scm = df_results[df_results['scm_type'] == scm_type]
        
        if 'f1_directed' in df_scm.columns:
            f1_by_method = {
                method: df_scm[df_scm['method'] == method]['f1_directed'].dropna().values
                for method in df_scm['method'].unique()
            }
            
            plot_metrics_comparison(
                f1_by_method,
                metric_name='F1 Score (Directed)',
                save_path=output_path / f"f1_comparison_{scm_type}.png"
            )
    
    print("\n" + "="*80)
    print("EXPERIMENT A COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    return df_results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment A: Synthetic Benchmark Recovery')
    parser.add_argument('--num_datasets', type=int, default=100, help='Number of datasets')
    parser.add_argument('--n_samples', type=int, default=500, help='Samples per dataset')
    parser.add_argument('--n_features', type=int, default=20, help='Features per dataset')
    parser.add_argument('--edge_prob', type=float, default=0.3, help='Edge probability')
    parser.add_argument('--output_dir', type=str, default='results/exp_a', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 10 datasets')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick test mode (10 datasets)...")
        args.num_datasets = 10
    
    run_experiment_a(
        num_datasets=args.num_datasets,
        n_samples=args.n_samples,
        n_features=args.n_features,
        edge_prob=args.edge_prob,
        output_dir=args.output_dir,
        seed=args.seed
    )
