#!/usr/bin/env python3
"""
Experiment E: TabPFN-Enhanced Causal Discovery

This experiment explores how TabPFN's attention patterns and predictions
can enhance traditional causal discovery algorithms from causality-lab.

We compare:
1. Standard causality-lab algorithms (RAI, ICD, FCI)
2. TabPFN-enhanced versions using custom conditional independence tests
3. Attn-SCM (direct attention map decoding)

The goal is to show that TabPFN's learned representations can improve
causal discovery on downstream tabular tasks.
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time
import warnings

from attn_scm.core import AttnSCM
from attn_scm.attention import AttentionExtractor
from attn_scm.metrics import compute_graph_metrics
from baselines import CAUSALITY_LAB_AVAILABLE, TABPFN_CAUSALITY_ADAPTER_AVAILABLE
from utils import generate_synthetic_dataset, plot_metrics_comparison

if CAUSALITY_LAB_AVAILABLE:
    from causality_lab.learn_structure import LearnStructRAI, LearnStructICD
    from causality_lab.cond_indep_tests import CondIndepParCorr
    from causality_lab.data import Dataset

if TABPFN_CAUSALITY_ADAPTER_AVAILABLE:
    from baselines.tabpfn_causality_adapter import CondIndepTabPFN, CondIndepAttentionWeighted

from tabpfn import TabPFNClassifier

warnings.filterwarnings('ignore')


def extract_attention_matrix(X: np.ndarray, y: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """
    Extract aggregated attention matrix from TabPFN.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,)
        device: Device for TabPFN

    Returns:
        Attention matrix (n_features, n_features)
    """
    model = TabPFNClassifier(device=device)
    extractor = AttentionExtractor(model)

    # Extract attention
    attention_layers = extractor.extract_attention(X, y)

    # Aggregate across layers and heads
    n_features = X.shape[1]
    attention_matrix = np.zeros((n_features, n_features))

    for layer_attention in attention_layers:
        # Average over batch and heads
        agg = layer_attention.mean(axis=(0, 1))
        # Extract feature-to-feature attention
        if agg.shape[0] >= n_features and agg.shape[1] >= n_features:
            attention_matrix += agg[:n_features, :n_features]

    # Normalize
    attention_matrix /= len(attention_layers)

    return attention_matrix


def run_rai_standard(X, feature_names, alpha=0.05):
    """Standard RAI with partial correlation test."""
    dataset = Dataset(X, var_names=feature_names)
    cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    nodes_set = set(feature_names)

    rai_learner = LearnStructRAI(nodes_set, cond_indep_test)
    rai_learner.learn_structure()

    # Convert to adjacency
    n_features = len(feature_names)
    adjacency = np.zeros((n_features, n_features), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    for edge in rai_learner.graph.edges:
        if hasattr(edge, 'source') and hasattr(edge, 'target'):
            source_idx = name_to_idx.get(edge.source.name)
            target_idx = name_to_idx.get(edge.target.name)
            if source_idx is not None and target_idx is not None:
                adjacency[source_idx, target_idx] = 1

    return adjacency


def run_rai_tabpfn_enhanced(X, y, feature_names, alpha=0.05, device='cpu'):
    """RAI enhanced with TabPFN conditional independence test."""
    dataset = Dataset(X, var_names=feature_names)
    cond_indep_test = CondIndepTabPFN(dataset, threshold=alpha, device=device)
    nodes_set = set(feature_names)

    rai_learner = LearnStructRAI(nodes_set, cond_indep_test)
    rai_learner.learn_structure()

    # Convert to adjacency
    n_features = len(feature_names)
    adjacency = np.zeros((n_features, n_features), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    for edge in rai_learner.graph.edges:
        if hasattr(edge, 'source') and hasattr(edge, 'target'):
            source_idx = name_to_idx.get(edge.source.name)
            target_idx = name_to_idx.get(edge.target.name)
            if source_idx is not None and target_idx is not None:
                adjacency[source_idx, target_idx] = 1

    return adjacency


def run_rai_attention_weighted(X, y, feature_names, attention_matrix, alpha=0.05):
    """RAI weighted by TabPFN attention patterns."""
    dataset = Dataset(X, var_names=feature_names)
    cond_indep_test = CondIndepAttentionWeighted(
        dataset,
        threshold=alpha,
        attention_weights=attention_matrix
    )
    nodes_set = set(feature_names)

    rai_learner = LearnStructRAI(nodes_set, cond_indep_test)
    rai_learner.learn_structure()

    # Convert to adjacency
    n_features = len(feature_names)
    adjacency = np.zeros((n_features, n_features), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    for edge in rai_learner.graph.edges:
        if hasattr(edge, 'source') and hasattr(edge, 'target'):
            source_idx = name_to_idx.get(edge.source.name)
            target_idx = name_to_idx.get(edge.target.name)
            if source_idx is not None and target_idx is not None:
                adjacency[source_idx, target_idx] = 1

    return adjacency


def run_experiment_tabpfn_causality(
    num_datasets: int = 50,
    n_samples: int = 500,
    n_features: int = 15,
    edge_prob: float = 0.3,
    scm_types: list = None,
    output_dir: str = "results/exp_e_tabpfn_causality",
    seed: int = 42
):
    """
    Run Experiment E: TabPFN-Enhanced Causal Discovery.

    Args:
        num_datasets: Number of synthetic datasets to generate
        n_samples: Number of samples per dataset
        n_features: Number of features per dataset (keep ≤20 for speed)
        edge_prob: Probability of edge in ground-truth DAG
        scm_types: List of SCM types to test
        output_dir: Directory to save results
        seed: Random seed
    """
    if scm_types is None:
        scm_types = ['linear_gaussian', 'nonlinear_anm']

    # Check dependencies
    if not CAUSALITY_LAB_AVAILABLE:
        print("ERROR: causality-lab is required for this experiment.")
        print("Install with: pip install git+https://github.com/IntelLabs/causality-lab.git")
        return None, None

    if not TABPFN_CAUSALITY_ADAPTER_AVAILABLE:
        print("ERROR: TabPFN causality adapter not available.")
        return None, None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENT E: TabPFN-Enhanced Causal Discovery")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of datasets: {num_datasets}")
    print(f"  - Samples per dataset: {n_samples}")
    print(f"  - Features per dataset: {n_features}")
    print(f"  - SCM types: {scm_types}")
    print("="*80)

    # Methods to test
    methods = [
        'RAI-Standard',
        'RAI-TabPFN',
        'RAI-Attention',
        'AttnSCM'
    ]

    print(f"\nMethods to test: {methods}")
    print("-"*80)

    # Results storage
    all_results = []

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

            feature_names = [f"X{i}" for i in range(n_features)]

            # Extract attention matrix once for reuse
            attention_matrix = None

            # Test each method
            for method_name in methods:
                try:
                    start_time = time.time()

                    if method_name == 'RAI-Standard':
                        pred_adj = run_rai_standard(X, feature_names, alpha=0.05)

                    elif method_name == 'RAI-TabPFN':
                        pred_adj = run_rai_tabpfn_enhanced(X, y, feature_names, alpha=0.05)

                    elif method_name == 'RAI-Attention':
                        if attention_matrix is None:
                            attention_matrix = extract_attention_matrix(X, y)
                        pred_adj = run_rai_attention_weighted(
                            X, y, feature_names, attention_matrix, alpha=0.05
                        )

                    elif method_name == 'AttnSCM':
                        model = AttnSCM(
                            top_k_heads=5,
                            threshold_method='otsu',
                            device='cpu'
                        )
                        pred_adj = model.fit(X, y)

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

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")

    # Print key findings
    if 'f1_directed' in df_results.columns:
        print("\n=== KEY FINDINGS ===")
        print("\nMethod Ranking by F1 Score (Directed):")
        f1_means = df_results.groupby('method')['f1_directed'].mean().sort_values(ascending=False)
        for rank, (method, score) in enumerate(f1_means.items(), 1):
            print(f"  {rank}. {method}: {score:.3f}")

        # Compare TabPFN-enhanced vs standard
        if 'RAI-Standard' in f1_means.index and 'RAI-TabPFN' in f1_means.index:
            improvement = f1_means['RAI-TabPFN'] - f1_means['RAI-Standard']
            print(f"\nRAI-TabPFN improvement over RAI-Standard: {improvement:+.3f}")

        if 'RAI-Standard' in f1_means.index and 'RAI-Attention' in f1_means.index:
            improvement = f1_means['RAI-Attention'] - f1_means['RAI-Standard']
            print(f"RAI-Attention improvement over RAI-Standard: {improvement:+.3f}")

    return df_results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment E: TabPFN-Enhanced Causal Discovery')
    parser.add_argument('--num_datasets', type=int, default=50, help='Number of datasets')
    parser.add_argument('--n_samples', type=int, default=500, help='Samples per dataset')
    parser.add_argument('--n_features', type=int, default=15, help='Features per dataset')
    parser.add_argument('--edge_prob', type=float, default=0.3, help='Edge probability')
    parser.add_argument('--output_dir', type=str, default='results/exp_e_tabpfn_causality', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 10 datasets')

    args = parser.parse_args()

    if args.quick_test:
        print("Running quick test mode (10 datasets, linear_gaussian only)...")
        args.num_datasets = 10
        args.n_features = 10
        scm_types = ['linear_gaussian']
    else:
        scm_types = None

    run_experiment_tabpfn_causality(
        num_datasets=args.num_datasets,
        n_samples=args.n_samples,
        n_features=args.n_features,
        edge_prob=args.edge_prob,
        scm_types=scm_types,
        output_dir=args.output_dir,
        seed=args.seed
    )
