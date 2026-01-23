import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from utils.visualization import plot_metrics_comparison

def plot_results(results_path="results/exp_a/raw_results.csv", output_dir="results/exp_a"):
    results_path = Path(results_path)
    output_path = Path(output_dir)
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    print(f"Loaded results from {results_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    scm_types = df['scm_type'].unique()
    
    for scm_type in scm_types:
        print(f"\nProcessing SCM type: {scm_type}")
        df_scm = df[df['scm_type'] == scm_type]
        
        # Plot SHD
        if 'shd' in df_scm.columns:
            shd_by_method = {
                method: df_scm[df_scm['method'] == method]['shd'].dropna().values
                for method in df_scm['method'].unique()
            }
            save_path = output_path / f"shd_comparison_{scm_type}.png"
            plot_metrics_comparison(
                shd_by_method, 
                metric_name='SHD',
                save_path=save_path
            )
            print(f"Saved SHD comparison to {save_path}")

        # Plot F1
        if 'f1_directed' in df_scm.columns:
            f1_by_method = {
                method: df_scm[df_scm['method'] == method]['f1_directed'].dropna().values
                for method in df_scm['method'].unique()
            }
            save_path = output_path / f"f1_comparison_{scm_type}.png"
            plot_metrics_comparison(
                f1_by_method, 
                metric_name='F1 Score (Directed)',
                save_path=save_path
            )
            print(f"Saved F1 comparison to {save_path}")

if __name__ == "__main__":
    plot_results()
