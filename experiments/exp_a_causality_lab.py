#!/usr/bin/env python3
"""
Experiment A: Synthetic Benchmark with Causality Lab Algorithms

Extended version of Experiment A that includes Intel Labs causality-lab algorithms.

Tests topological accuracy of:
- Attn-SCM (our method)
- PC Algorithm (causal-learn)
- NOTEARS
- RAI (Causality Lab)
- FCI (Causality Lab)
- ICD (Causality Lab)

Metrics: Structural Hamming Distance (SHD), F1-Score
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time

from attn_scm.core import AttnSCM
from attn_scm.metrics import compute_graph_metrics
from baselines import (
    run_pc_algorithm,
    run_notears,
    run_random_baseline,
    CAUSALITY_LAB_AVAILABLE
)
from utils import generate_synthetic_dataset, save_results, plot_metrics_comparison

if CAUSALITY_LAB_AVAILABLE:
    from baselines import run_rai, run_fci, run_icd


def run_experiment_a_causality_lab(
    num_datasets: int = 100,
    n_samples: int = 500,
    n_features: int = 20,
    edge_prob: float = 0.3,
    scm_types: list = None,
    output_dir: str = "results/exp_a_causality_lab",
    seed: int = 42,
    include_causality_lab: bool = True
):
    """
    Run Experiment A with Causality Lab algorithms.

    Args:
        num_datasets: Number of synthetic datasets to generate
        n_samples: Number of samples per dataset
        n_features: Number of features per dataset
        edge_prob: Probability of edge in ground-truth DAG
        scm_types: List of SCM types to test
        output_dir: Directory to save results
        seed: Random seed
        include_causality_lab: Whether to include causality-lab algorithms
    """
    if scm_types is None:
        scm_types = ['linear_gaussian', 'linear_nongaussian', 'nonlinear_anm']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENT A: Synthetic Benchmark with Causality Lab")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of datasets: {num_datasets}")
    print(f"  - Samples per dataset: {n_samples}")
    print(f"  - Features per dataset: {n_features}")
    print(f"  - SCM types: {scm_types}")
    print(f"  - Causality Lab available: {CAUSALITY_LAB_AVAILABLE}")
    print(f"  - Include Causality Lab: {include_causality_lab and CAUSALITY_LAB_AVAILABLE}")
    print("="*80)

    # Results storage
    all_results = []

    # Define methods to test
    methods = {
        'AttnSCM': True,
        'PC': True,
        'NOTEARS': True,
        'Random': True
    }

    # Add causality-lab methods if available
    if include_causality_lab and CAUSALITY_LAB_AVAILABLE:
        methods.update({
            'RAI': True,
            'FCI': True,
            'ICD': True
        })

    print(f"\nMethods to test: {list(methods.keys())}")
    print("-"*80)

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

            # Feature names for causality-lab
            feature_names = [f"X{i}" for i in range(n_features)]

            # Test each method
            for method_name in methods:
                try:
                    start_time = time.time()

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

                    elif method_name == 'RAI':
                        pred_adj = run_rai(
                            X,
                            alpha=0.05,
                            indep_test='parcorr',
                            feature_names=feature_names
                        )

                    elif method_name == 'FCI':
                        pred_adj = run_fci(
                            X,
                            alpha=0.05,
                            indep_test='parcorr',
                            feature_names=feature_names
                        )

                    elif method_name == 'ICD':
                        pred_adj = run_icd(
                            X,
                            alpha=0.05,
                            indep_test='parcorr',
                            feature_names=feature_names
                        )

                    else:
                        continue

                    elapsed_time = time.time() - start_time

                    # Compute metrics
                    metrics = compute_graph_metrics(pred_adj, true_adj)

                    # Store results
                    result = {
                        'scm_type': scm_type,
                        'dataset_idx': dataset_idx,
                        'method': method_name,
                        'elapsed_time': elapsed_time,
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
    metrics_to_summarize = ['shd', 'f1_directed', 'f1_undirected', 'precision', 'recall', 'elapsed_time']
    agg_dict = {m: ['mean', 'std', 'median'] for m in metrics_to_summarize if m in df_results.columns}

    summary = df_results.groupby(['scm_type', 'method']).agg(agg_dict).round(3)

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

    # SHD comparison by SCM type
    for scm_type in scm_types:
        df_scm = df_results[df_results['scm_type'] == scm_type]

        if 'shd' in df_scm.columns:
            shd_by_method = {
                method: df_scm[df_scm['method'] == method]['shd'].dropna().values
                for method in df_scm['method'].unique()
            }

            if any(len(v) > 0 for v in shd_by_method.values()):
                plot_metrics_comparison(
                    shd_by_method,
                    metric_name='SHD',
                    save_path=output_path / f"shd_comparison_{scm_type}.png"
                )

    # F1 comparison by SCM type
    for scm_type in scm_types:
        df_scm = df_results[df_results['scm_type'] == scm_type]

        if 'f1_directed' in df_scm.columns:
            f1_by_method = {
                method: df_scm[df_scm['method'] == method]['f1_directed'].dropna().values
                for method in df_scm['method'].unique()
            }

            if any(len(v) > 0 for v in f1_by_method.values()):
                plot_metrics_comparison(
                    f1_by_method,
                    metric_name='F1 Score (Directed)',
                    save_path=output_path / f"f1_comparison_{scm_type}.png"
                )

    # Timing comparison (aggregate across all SCM types)
    if 'elapsed_time' in df_results.columns:
        timing_by_method = {
            method: df_results[df_results['method'] == method]['elapsed_time'].dropna().values
            for method in df_results['method'].unique()
        }

        if any(len(v) > 0 for v in timing_by_method.values()):
            plot_metrics_comparison(
                timing_by_method,
                metric_name='Elapsed Time (seconds)',
                save_path=output_path / "timing_comparison.png"
            )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")

    # Print method ranking by F1 score
    if 'f1_directed' in df_results.columns:
        print("\nMethod Ranking by F1 Score (Directed):")
        f1_means = df_results.groupby('method')['f1_directed'].mean().sort_values(ascending=False)
        for rank, (method, score) in enumerate(f1_means.items(), 1):
            print(f"  {rank}. {method}: {score:.3f}")

    return df_results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment A with Causality Lab algorithms')
    parser.add_argument('--num_datasets', type=int, default=100, help='Number of datasets')
    parser.add_argument('--n_samples', type=int, default=500, help='Samples per dataset')
    parser.add_argument('--n_features', type=int, default=20, help='Features per dataset')
    parser.add_argument('--edge_prob', type=float, default=0.3, help='Edge probability')
    parser.add_argument('--output_dir', type=str, default='results/exp_a_causality_lab', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 10 datasets')
    parser.add_argument('--no_causality_lab', action='store_true', help='Skip causality-lab algorithms')

    args = parser.parse_args()

    if args.quick_test:
        print("Running quick test mode (10 datasets, linear_gaussian only)...")
        args.num_datasets = 10
        scm_types = ['linear_gaussian']
    else:
        scm_types = None

    if not CAUSALITY_LAB_AVAILABLE and not args.no_causality_lab:
        print("\nWARNING: causality-lab not available. Install with:")
        print("  pip install git+https://github.com/IntelLabs/causality-lab.git")
        print("Running without causality-lab algorithms...\n")

    run_experiment_a_causality_lab(
        num_datasets=args.num_datasets,
        n_samples=args.n_samples,
        n_features=args.n_features,
        edge_prob=args.edge_prob,
        scm_types=scm_types,
        output_dir=args.output_dir,
        seed=args.seed,
        include_causality_lab=not args.no_causality_lab
    )
