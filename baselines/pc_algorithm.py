"""PC Algorithm baseline using causal-learn library."""

import numpy as np
from typing import Optional

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False


def run_pc_algorithm(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'fisherz',
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2
) -> np.ndarray:
    """
    Run PC algorithm for causal discovery.
    
    The PC algorithm uses conditional independence tests to learn causal structure.
    
    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('fisherz', 'chisq', 'gsq')
        stable: Whether to use stable-PC variant
        uc_rule: Rule for orientation (0, 1, 2, 3)
        uc_priority: Priority for unshielded colliders
        
    Returns:
        Adjacency matrix (n_features, n_features)
    """
    if not CAUSAL_LEARN_AVAILABLE:
        raise ImportError("causal-learn is required. Install with: pip install causal-learn")
    
    # Run PC algorithm
    cg = pc(
        X,
        alpha=alpha,
        indep_test=indep_test,
        stable=stable,
        uc_rule=uc_rule,
        uc_priority=uc_priority
    )
    
    # Extract adjacency matrix from CausalGraph object
    # PC returns a CPDAG (completed partially directed acyclic graph)
    adjacency = cg.G.graph
    
    # Convert to binary directed adjacency
    # In causal-learn: -1 means edge exists, 1 means directed edge, 0 means no edge
    # We want: 1 if i->j exists, 0 otherwise
    n_features = adjacency.shape[0]
    directed_adj = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            # Check if there's a directed edge from i to j
            # In the matrix: adjacency[i,j] = -1 and adjacency[j,i] = 1 means i->j
            if adjacency[j, i] == -1 and adjacency[i, j] == 1:
                directed_adj[i, j] = 1
            # Undirected edges: both are -1
            elif adjacency[i, j] == -1 and adjacency[j, i] == -1:
                # For undirected, we arbitrarily orient based on index
                if i < j:
                    directed_adj[i, j] = 1
    
    return directed_adj
