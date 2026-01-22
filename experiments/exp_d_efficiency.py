#!/usr/bin/env python3
"""
Experiment D: Computational Efficiency Profile

Compares wall-clock time of Attn-SCM vs traditional causal discovery methods.

Expected: Constant/linear scaling for Attn-SCM vs exponential for PC
"""

import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import time
import matplotlib.pyplot as plt

from attn_scm.core import AttnSCM
from baselines import run_pc_algorithm, run_notears
from utils import generate_synthetic_dataset, save_results


def run_experiment_d(
    max_n: int = 1000,
    max_d: int = 50,
    n_trials: int = 5,
    output_dir: str = "results/exp_d",
    seed: int = 42
):
    """
    Run Experiment D: Computational Efficiency Profiling.
    
    Args:
        max_n: Maximum number of samples
        max_d: Maximum number of features
        n_trials: Number of trials per configuration
        output_dir: Output directory
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXPERIMENT D: Computational Efficiency Profile")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Max samples: {max_n}")
    print(f"  - Max features: {max_d}")
    print(f"  - Trials per config: {n_trials}")
    print("="*80)
    
    all_results = []
    
    # Test 1: Scaling with number of samples (N)
    print("\nTest 1: Scaling with Number of Samples")
    print("-"*80)
    
    sample_sizes = [100, 200, 500, 1000] if max_n >= 1000 else [100, 200, 500]
    d_fixed = 20
    
    for n in sample_sizes:
        print(f"\nTesting N={n}, d={d_fixed}")
        
        for method_name in ['AttnSCM', 'PC', 'NOTEARS']:
            times = []
            
            for trial in range(n_trials):
                # Generate data
                X, y, true_adj = generate_synthetic_dataset(
                    n_nodes=d_fixed,
                    n_samples=n,
                    edge_prob=0.3,
                    scm_type='linear_gaussian',
                    seed=seed + trial
                )
                
                # Time the method
                start_time = time.time()
                
                try:
                    if method_name == 'AttnSCM':
                        model = AttnSCM(top_k_heads=5, device='cpu')
                        _ = model.fit(X, y)
                    
                    elif method_name == 'PC':
                        _ = run_pc_algorithm(X, alpha=0.05)
                    
                    elif method_name == 'NOTEARS':
                        _ = run_notears(X, lambda1=0.1)
                    
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)
                
                except Exception as e:
                    print(f"  Error with {method_name}: {e}")
                    times.append(np.nan)
            
            # Record results
            mean_time = np.nanmean(times)
            std_time = np.nanstd(times)
            
            print(f"  {method_name}: {mean_time:.3f}s ± {std_time:.3f}s")
            
            all_results.append({
                'test': 'scaling_n',
                'n': n,
                'd': d_fixed,
                'method': method_name,
                'mean_time': mean_time,
                'std_time': std_time
            })
    
    # Test 2: Scaling with number of features (d)
    print("\n\nTest 2: Scaling with Number of Features")
    print("-"*80)
    
    feature_sizes = [10, 20, 30, 50] if max_d >= 50 else [10, 20, 30]
    n_fixed = 500
    
    for d in feature_sizes:
        print(f"\nTesting N={n_fixed}, d={d}")
        
        for method_name in ['AttnSCM', 'PC', 'NOTEARS']:
            times = []
            
            for trial in range(n_trials):
                # Generate data
                X, y, true_adj = generate_synthetic_dataset(
                    n_nodes=d,
                    n_samples=n_fixed,
                    edge_prob=0.3,
                    scm_type='linear_gaussian',
                    seed=seed + trial
                )
                
                start_time = time.time()
                
                try:
                    if method_name == 'AttnSCM':
                        model = AttnSCM(top_k_heads=5, device='cpu')
                        _ = model.fit(X, y)
                    
                    elif method_name == 'PC':
                        _ = run_pc_algorithm(X, alpha=0.05)
                    
                    elif method_name == 'NOTEARS':
                        _ = run_notears(X, lambda1=0.1)
                    
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)
                
                except Exception as e:
                    print(f"  Error with {method_name}: {e}")
                    times.append(np.nan)
            
            mean_time = np.nanmean(times)
            std_time = np.nanstd(times)
            
            print(f"  {method_name}: {mean_time:.3f}s ± {std_time:.3f}s")
            
            all_results.append({
                'test': 'scaling_d',
                'n': n_fixed,
                'd': d,
                'method': method_name,
                'mean_time': mean_time,
                'std_time': std_time
            })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    csv_path = output_path / "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Time vs N
    df_n = df_results[df_results['test'] == 'scaling_n']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in ['AttnSCM', 'PC', 'NOTEARS']:
        df_method = df_n[df_n['method'] == method]
        ax.plot(
            df_method['n'],
            df_method['mean_time'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=method
        )
    
    ax.set_xlabel('Number of Samples (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Time vs Sample Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "time_vs_n.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Time vs d
    df_d = df_results[df_results['test'] == 'scaling_d']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in ['AttnSCM', 'PC', 'NOTEARS']:
        df_method = df_d[df_d['method'] == method]
        ax.plot(
            df_method['d'],
            df_method['mean_time'],
            marker='s',
            linewidth=2,
            markersize=8,
            label=method
        )
    
    ax.set_xlabel('Number of Features (d)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Time vs Feature Dimension', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "time_vs_d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("EXPERIMENT D COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment D: Efficiency Profiling')
    parser.add_argument('--max_n', type=int, default=1000, help='Max number of samples')
    parser.add_argument('--max_d', type=int, default=50, help='Max number of features')
    parser.add_argument('--n_trials', type=int, default=5, help='Trials per config')
    parser.add_argument('--output_dir', type=str, default='results/exp_d', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_experiment_d(
        max_n=args.max_n,
        max_d=args.max_d,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed=args.seed
    )
