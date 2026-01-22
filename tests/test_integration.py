"""
Simple integration test for Attn-SCM pipeline.

Tests the complete pipeline on a synthetic dataset.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attn_scm.core import AttnSCM
from attn_scm.metrics import compute_graph_metrics
from utils import generate_synthetic_dataset


def test_attn_scm_pipeline():
    """Test complete Attn-SCM pipeline."""
    print("Testing Attn-SCM Pipeline")
    print("="*80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic dataset...")
    X, y, true_adj = generate_synthetic_dataset(
        n_nodes=10,
        n_samples=200,
        edge_prob=0.3,
        scm_type='linear_gaussian',
        seed=42
    )
    print(f"   Data shape: {X.shape}")
    print(f"   True graph edges: {(true_adj > 0).sum()}")
    
    # Initialize model
    print("\n2. Initializing Attn-SCM...")
    model = AttnSCM(
        top_k_heads=3,
        threshold_method='otsu',
        device='cpu'
    )
    print("   Model initialized")
    
    # Extract causal graph
    print("\n3. Extracting causal graph...")
    try:
        pred_adj = model.fit(X, y)
        print(f"   Predicted graph edges: {(pred_adj > 0).sum()}")
        
        # Compute metrics
        print("\n4. Computing metrics...")
        metrics = compute_graph_metrics(pred_adj, true_adj)
        print(f"   SHD: {metrics['shd']}")
        print(f"   F1 (directed): {metrics['f1_directed']:.3f}")
        print(f"   F1 (undirected): {metrics['f1_undirected']:.3f}")
        
        # Get Markov Blanket
        print("\n5. Extracting Markov Blanket of node 0...")
        mb = model.get_markov_blanket(0)
        print(f"   Markov Blanket: {mb}")
        
        print("\n" + "="*80)
        print("TEST PASSED")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        
        return False


if __name__ == '__main__':
    success = test_attn_scm_pipeline()
    sys.exit(0 if success else 1)
