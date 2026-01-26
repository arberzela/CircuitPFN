#!/usr/bin/env python3
"""
Experiment F: Real-World Tabular Datasets with Causality Lab

Apply causality-lab algorithms and AttnSCM to real-world tabular datasets
from OpenML to discover causal relationships in downstream TabPFN tasks.

Since ground truth causal graphs are unknown for real datasets, we evaluate:
1. Graph structure properties (sparsity, connectivity)
2. Consistency across methods
3. Predictive utility via Markov Blanket feature selection

Datasets tested:
- Credit-G (credit risk prediction)
- Diabetes (diabetes diagnosis)
- Blood Transfusion (donation prediction)
- Phoneme (phoneme classification)
"""

import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import time
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import openml

from attn_scm.core import AttnSCM
from attn_scm.graph import extract_markov_blanket
from baselines import CAUSALITY_LAB_AVAILABLE
from utils import plot_metrics_comparison

if CAUSALITY_LAB_AVAILABLE:
    from baselines import run_rai, run_fci, run_icd

warnings.filterwarnings('ignore')


# Dataset IDs from OpenML
DATASET_CONFIG = {
    'credit-g': {'id': 31, 'name': 'Credit Approval'},
    'diabetes': {'id': 37, 'name': 'Pima Indians Diabetes'},
    'blood-transfusion': {'id': 1464, 'name': 'Blood Transfusion'},
    'phoneme': {'id': 1489, 'name': 'Phoneme Classification'},
}


def load_openml_dataset(dataset_key: str):
    """
    Load and preprocess OpenML dataset.

    Args:
        dataset_key: Dataset key from DATASET_CONFIG

    Returns:
        Tuple of (X, y, feature_names)
    """
    dataset_id = DATASET_CONFIG[dataset_key]['id']
    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    # Handle categorical variables
    if categorical_indicator is not None:
        for col_idx, is_categorical in enumerate(categorical_indicator):
            if is_categorical:
                le = LabelEncoder()
                X[:, col_idx] = le.fit_transform(X[:, col_idx].astype(str))

    # Handle missing values
    if np.any(np.isnan(X)):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

    # Encode labels
    if y.dtype == object or len(np.unique(y)) > 10:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Make binary if needed
    if len(np.unique(y)) > 2:
        y = (y == np.unique(y)[0]).astype(int)

    # Limit to TabPFN constraints
    if X.shape[0] > 1000:
        X = X[:1000]
        y = y[:1000]
    if X.shape[1] > 100:
        X = X[:, :100]
        feature_names = feature_names[:100]

    return X, y, list(feature_names)


def evaluate_markov_blanket_features(
    X: np.ndarray,
    y: np.ndarray,
    adjacency: np.ndarray,
    target_idx: int = -1
) -> dict:
    """
    Evaluate predictive utility of Markov Blanket features.

    Args:
        X: Features
        y: Target labels
        adjacency: Predicted causal adjacency matrix
        target_idx: Index of target variable (default: use y as separate target)

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract Markov Blanket (parents + children + spouses)
    mb_indices = extract_markov_blanket(adjacency, target_idx)

    if len(mb_indices) == 0:
        return {
            'mb_size': 0,
            'mb_cv_score': 0.0,
            'all_features_cv_score': 0.0,
            'mb_improvement': 0.0
        }

    # Evaluate with Markov Blanket features only
    X_mb = X[:, list(mb_indices)]
    clf_mb = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    scores_mb = cross_val_score(clf_mb, X_mb, y, cv=3, scoring='accuracy')
    mb_cv_score = scores_mb.mean()

    # Evaluate with all features
    clf_all = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    scores_all = cross_val_score(clf_all, X, y, cv=3, scoring='accuracy')
    all_cv_score = scores_all.mean()

    return {
        'mb_size': len(mb_indices),
        'mb_cv_score': mb_cv_score,
        'all_features_cv_score': all_cv_score,
        'mb_improvement': mb_cv_score - all_cv_score
    }


def analyze_graph_structure(adjacency: np.ndarray) -> dict:
    """
    Analyze structural properties of discovered graph.

    Args:
        adjacency: Binary adjacency matrix

    Returns:
        Dictionary of graph properties
    """
    n_nodes = adjacency.shape[0]
    n_edges = np.sum(adjacency)
    sparsity = 1.0 - (n_edges / (n_nodes * (n_nodes - 1)))

    # In-degree and out-degree statistics
    in_degrees = adjacency.sum(axis=0)
    out_degrees = adjacency.sum(axis=1)

    return {
        'n_nodes': n_nodes,
        'n_edges': int(n_edges),
        'sparsity': sparsity,
        'avg_in_degree': in_degrees.mean(),
        'max_in_degree': int(in_degrees.max()),
        'avg_out_degree': out_degrees.mean(),
        'max_out_degree': int(out_degrees.max())
    }


def run_experiment_real_datasets(
    datasets: list = None,
    output_dir: str = "results/exp_f_real_datasets",
    alpha: float = 0.05
):
    """
    Run causal discovery on real-world tabular datasets.

    Args:
        datasets: List of dataset keys to test (default: all)
        output_dir: Directory to save results
        alpha: Significance level for causality-lab methods
    """
    if datasets is None:
        datasets = list(DATASET_CONFIG.keys())

    if not CAUSALITY_LAB_AVAILABLE:
        print("WARNING: causality-lab not available. Only running AttnSCM.")
        print("Install with: pip install git+https://github.com/IntelLabs/causality-lab.git\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENT F: Real-World Tabular Datasets")
    print("="*80)
    print(f"Datasets: {datasets}")
    print("="*80)

    # Methods to test
    methods = ['AttnSCM']
    if CAUSALITY_LAB_AVAILABLE:
        methods.extend(['RAI', 'ICD', 'FCI'])

    print(f"Methods: {methods}\n")

    # Results storage
    all_results = []
    all_graphs = {}

    # Test each dataset
    for dataset_key in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {DATASET_CONFIG[dataset_key]['name']} ({dataset_key})")
        print('='*80)

        try:
            # Load dataset
            X, y, feature_names = load_openml_dataset(dataset_key)
            print(f"Shape: {X.shape}, Features: {len(feature_names)}")

            dataset_graphs = {}

            # Test each method
            for method_name in methods:
                print(f"\n  Testing {method_name}...")

                try:
                    start_time = time.time()

                    # Run causal discovery
                    if method_name == 'AttnSCM':
                        model = AttnSCM(
                            top_k_heads=5,
                            threshold_method='otsu',
                            device='cpu'
                        )
                        pred_adj = model.fit(X, y)

                    elif method_name == 'RAI':
                        pred_adj = run_rai(X, alpha=alpha, feature_names=feature_names)

                    elif method_name == 'ICD':
                        pred_adj = run_icd(X, alpha=alpha, feature_names=feature_names)

                    elif method_name == 'FCI':
                        pred_adj = run_fci(X, alpha=alpha, feature_names=feature_names)

                    else:
                        continue

                    elapsed_time = time.time() - start_time

                    # Analyze graph structure
                    graph_props = analyze_graph_structure(pred_adj)

                    # Evaluate Markov Blanket for target prediction
                    # Use target as a synthetic last variable
                    mb_eval = evaluate_markov_blanket_features(X, y, pred_adj, target_idx=0)

                    # Store results
                    result = {
                        'dataset': dataset_key,
                        'dataset_name': DATASET_CONFIG[dataset_key]['name'],
                        'method': method_name,
                        'elapsed_time': elapsed_time,
                        **graph_props,
                        **mb_eval
                    }
                    all_results.append(result)

                    # Save graph
                    dataset_graphs[method_name] = pred_adj

                    print(f"    Edges: {graph_props['n_edges']}, "
                          f"Sparsity: {graph_props['sparsity']:.2f}, "
                          f"Time: {elapsed_time:.2f}s")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results.append({
                        'dataset': dataset_key,
                        'dataset_name': DATASET_CONFIG[dataset_key]['name'],
                        'method': method_name,
                        'error': str(e)
                    })

            all_graphs[dataset_key] = dataset_graphs

        except Exception as e:
            print(f"ERROR loading dataset {dataset_key}: {e}")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save results
    csv_path = output_path / "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Saved results to: {csv_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY METHOD")
    print("="*80)

    summary_cols = ['n_edges', 'sparsity', 'mb_size', 'mb_cv_score', 'elapsed_time']
    summary_cols = [c for c in summary_cols if c in df_results.columns]

    if summary_cols:
        summary = df_results.groupby('method')[summary_cols].agg(['mean', 'std']).round(3)
        print(summary)

        summary_path = output_path / "summary.csv"
        summary.to_csv(summary_path)

    # Per-dataset summary
    print("\n" + "="*80)
    print("SUMMARY BY DATASET")
    print("="*80)

    if summary_cols:
        dataset_summary = df_results.groupby('dataset')[summary_cols].agg(['mean', 'std']).round(3)
        print(dataset_summary)

    # Analyze method agreement
    print("\n" + "="*80)
    print("METHOD AGREEMENT ANALYSIS")
    print("="*80)

    for dataset_key in datasets:
        if dataset_key not in all_graphs:
            continue

        graphs = all_graphs[dataset_key]
        if len(graphs) < 2:
            continue

        print(f"\nDataset: {dataset_key}")

        method_pairs = []
        for i, method1 in enumerate(methods):
            if method1 not in graphs:
                continue
            for method2 in methods[i+1:]:
                if method2 not in graphs:
                    continue

                adj1 = graphs[method1]
                adj2 = graphs[method2]

                # Compute edge agreement
                agreement = np.mean(adj1 == adj2)
                n_common_edges = np.sum((adj1 == 1) & (adj2 == 1))

                print(f"  {method1} vs {method2}: "
                      f"Agreement={agreement:.2f}, Common edges={n_common_edges}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")

    return df_results, all_graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment F: Real-World Datasets')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIG.keys()),
                        help='Datasets to test (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/exp_f_real_datasets',
                        help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for causality-lab methods')

    args = parser.parse_args()

    run_experiment_real_datasets(
        datasets=args.datasets,
        output_dir=args.output_dir,
        alpha=args.alpha
    )
