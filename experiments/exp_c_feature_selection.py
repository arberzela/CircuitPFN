#!/usr/bin/env python3
"""
Experiment C: Downstream Zero-Shot Feature Selection

Demonstrates practical utility of Attn-SCM via Markov Blanket-based feature selection.

Compares:
- Baseline: XGBoost on all features
- Attn-SCM: XGBoost on Markov Blanket features only

Metrics: AUC, Feature Reduction Ratio (FRR)
"""

import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb

try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False

from attn_scm.core import AttnSCM
from utils import save_results


# Selected OpenML datasets with known characteristics
DATASET_IDS = {
    'credit-g': 31,         # German Credit
    'phoneme': 1489,        # Phoneme
    'blood-transfusion': 1464,
    'diabetes': 37,         # Pima Indians Diabetes
    'kr-vs-kp': 3            # King-Rook vs King-Pawn
}


def load_openml_dataset(dataset_name: str):
    """Load dataset from OpenML."""
    if not OPENML_AVAILABLE:
        raise ImportError("OpenML is required. Install with: pip install openml")
    
    dataset_id = DATASET_IDS.get(dataset_name)
    if dataset_id is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical, feature_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    
    # Handle categorical variables (simple encoding)
    if categorical is not None and len(categorical) > 0:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical:
            if col < X.shape[1]:
                X[:, col] = le.fit_transform(X[:, col].astype(str))
    
    # Handle binary classification
    if len(np.unique(y)) > 2:
        # Multi-class: convert to binary (class 0 vs rest)
        y = (y == np.unique(y)[0]).astype(int)
    
    return X, y, feature_names


def run_experiment_c(
    datasets: list = None,
    output_dir: str = "results/exp_c",
    n_cv_folds: int = 5,
    seed: int = 42
):
    """
    Run Experiment C: Feature Selection via Markov Blanket.
    
    Args:
        datasets: List of dataset names to test
        output_dir: Output directory
        n_cv_folds: Number of cross-validation folds
        seed: Random seed
    """
    if datasets is None:
        datasets = ['credit-g', 'phoneme', 'blood-transfusion', 'diabetes', 'kr-vs-kp']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXPERIMENT C: Downstream Zero-Shot Feature Selection")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"CV Folds: {n_cv_folds}")
    print("="*80)
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset_name}")
        print('='*80)
        
        try:
            # Load dataset
            print("Loading data...")
            X, y, feature_names = load_openml_dataset(dataset_name)
            
            n_samples, n_features = X.shape
            print(f"Loaded: {n_samples} samples, {n_features} features")
            
            # Ensure data is numeric and handle NaNs
            X = np.nan_to_num(X, nan=0.0)
            
            # Limit features for TabPFN
            if n_features > 100:
                print(f"Reducing features from {n_features} to 100 (TabPFN limit)")
                X = X[:, :100]
                n_features = 100
            
            # Limit samples for TabPFN
            if n_samples > 1000:
                print(f"Sampling 1000 samples (TabPFN limit)")
                np.random.seed(seed)
                sample_indices = np.random.choice(n_samples, 1000, replace=False)
                X = X[sample_indices]
                y = y[sample_indices]
                n_samples = 1000
            
            # Baseline: XGBoost on all features
            print("\n1. Baseline: XGBoost on ALL features")
            print("-"*80)
            
            xgb_baseline = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=seed,
                eval_metric='logloss'
            )
            
            baseline_scores = cross_val_score(
                xgb_baseline, X, y,
                cv=n_cv_folds,
                scoring='roc_auc'
            )
            baseline_auc = np.mean(baseline_scores)
            baseline_auc_std = np.std(baseline_scores)
            
            print(f"Baseline AUC: {baseline_auc:.4f} ± {baseline_auc_std:.4f}")
            
            # Attn-SCM: Extract Markov Blanket
            print("\n2. Attn-SCM: Extracting Markov Blanket")
            print("-"*80)
            
            model_scm = AttnSCM(
                top_k_heads=5,
                threshold_method='otsu',
                device='cpu'
            )
            
            # Fit Attn-SCM
            adjacency = model_scm.fit(X, y)
            
            # Target is typically the last feature or a synthetic constructed one
            # For this experiment, we'll create a synthetic target node
            # and extract its Markov Blanket from the feature graph
            
            # Get Markov Blanket of most connected node (heuristic)
            node_degrees = adjacency.sum(axis=0) + adjacency.sum(axis=1)
            most_connected_node = np.argmax(node_degrees)
            
            mb_features = model_scm.get_markov_blanket(most_connected_node)
            
            if len(mb_features) == 0:
                print("Warning: Empty Markov Blanket! Using top 5 most connected nodes")
                mb_features = np.argsort(node_degrees)[-5:].tolist()
            
            frr = 1.0 - (len(mb_features) / n_features)
            
            print(f"Markov Blanket size: {len(mb_features)} / {n_features} features")
            print(f"Feature Reduction Ratio: {frr:.2%}")
            print(f"Selected features: {mb_features[:10]}...")  # Show first 10
            
            # XGBoost on Markov Blanket features
            print("\n3. XGBoost on Markov Blanket features")
            print("-"*80)
            
            if len(mb_features) > 0:
                X_mb = X[:, mb_features]
                
                xgb_mb = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=seed,
                    eval_metric='logloss'
                )
                
                mb_scores = cross_val_score(
                    xgb_mb, X_mb, y,
                    cv=n_cv_folds,
                    scoring='roc_auc'
                )
                mb_auc = np.mean(mb_scores)
                mb_auc_std = np.std(mb_scores)
                
                print(f"Attn-SCM AUC: {mb_auc:.4f} ± {mb_auc_std:.4f}")
                print(f"Δ AUC: {mb_auc - baseline_auc:+.4f}")
            else:
                mb_auc = 0.0
                mb_auc_std = 0.0
            
            # Store results
            result = {
                'dataset': dataset_name,
                'n_samples': n_samples,
                'n_features': n_features,
                'baseline_auc': baseline_auc,
                'baseline_auc_std': baseline_auc_std,
                'attnscm_auc': mb_auc,
                'attnscm_auc_std': mb_auc_std,
                'delta_auc': mb_auc - baseline_auc,
                'mb_size': len(mb_features),
                'feature_reduction_ratio': frr
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"\nError with dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'dataset': dataset_name,
                'error': str(e)
            })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    csv_path = output_path / "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print('='*80)
    print(df_results[['dataset', 'baseline_auc', 'attnscm_auc', 'delta_auc', 'feature_reduction_ratio']])
    
    print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT C COMPLETE")
    print("="*80)
    
    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment C: Feature Selection')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['credit-g', 'phoneme', 'blood-transfusion', 'diabetes', 'kr-vs-kp'],
        help='Dataset names'
    )
    parser.add_argument('--output_dir', type=str, default='results/exp_c', help='Output directory')
    parser.add_argument('--n_cv_folds', type=int, default=5, help='CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_experiment_c(
        datasets=args.datasets,
        output_dir=args.output_dir,
        n_cv_folds=args.n_cv_folds,
        seed=args.seed
    )
