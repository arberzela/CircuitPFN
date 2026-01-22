
import numpy as np
from attn_scm.core import AttnSCM
import traceback

def reproduce():
    print("Starting reproduction...")
    try:
        # Create minimal synthetic data
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = AttnSCM(top_k_heads=2, device='cpu', threshold_method='otsu')
        print("Fitting AttnSCM...")
        adj = model.fit(X, y)
        print("Fit successful!")
        print(adj)
        
    except Exception as e:
        print(f"Caught exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import networkx as nx
    print(f"NetworkX version: {nx.__version__}")
    print(f"Has from_numpy_array: {hasattr(nx, 'from_numpy_array')}")
    print(f"Has from_numpy_matrix: {hasattr(nx, 'from_numpy_matrix')}")
    reproduce()
