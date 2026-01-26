"""Causality Lab algorithms wrapper for Intel Labs causality-lab library.

This module provides wrappers for various causal discovery algorithms from Intel Labs:
- PC algorithm (alternative implementation to causal-learn)
- RAI (Recursive Autonomy Identification)
- B-RAI (Bootstrap/Bayesian RAI with uncertainty estimation)
- FCI (Fast Causal Inference - handles latent confounders)
- ICD (Iterative Causal Discovery - handles latent confounders)
"""

import numpy as np
from typing import Optional, Tuple, Union

try:
    from causality_lab.graph import Graph, EdgeType, UndirectedEdge, DirectedEdge
    from causality_lab.learn_structure import (
        LearnStructPC,
        LearnStructRAI,
        LearnStructBRAI,
        LearnStructFCI,
        LearnStructICD
    )
    from causality_lab.cond_indep_tests import CondIndepParCorr, CondIndepCMI
    from causality_lab.data import Dataset
    CAUSALITY_LAB_AVAILABLE = True
except ImportError:
    CAUSALITY_LAB_AVAILABLE = False


def _check_availability():
    """Check if causality-lab is available."""
    if not CAUSALITY_LAB_AVAILABLE:
        raise ImportError(
            "causality-lab is required. Install with:\n"
            "pip install git+https://github.com/IntelLabs/causality-lab.git"
        )


def _create_dataset(X: np.ndarray, feature_names: Optional[list] = None) -> 'Dataset':
    """
    Create a causality-lab Dataset object from numpy array.

    Args:
        X: Data matrix (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Dataset object for causality-lab
    """
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create Dataset object
    dataset = Dataset(X, var_names=feature_names)
    return dataset


def _graph_to_adjacency(graph: 'Graph', n_features: int, feature_names: list) -> np.ndarray:
    """
    Convert causality-lab Graph to adjacency matrix.

    Args:
        graph: causality-lab Graph object
        n_features: Number of features/nodes
        feature_names: List of feature names

    Returns:
        Binary directed adjacency matrix (n_features, n_features)
    """
    adjacency = np.zeros((n_features, n_features), dtype=int)

    # Create name to index mapping
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Iterate over edges in the graph
    for edge in graph.edges:
        if isinstance(edge, DirectedEdge):
            # Directed edge: source -> target
            source_idx = name_to_idx[edge.source.name]
            target_idx = name_to_idx[edge.target.name]
            adjacency[source_idx, target_idx] = 1
        elif isinstance(edge, UndirectedEdge):
            # Undirected edge: arbitrarily orient based on index
            node1_idx = name_to_idx[edge.node1.name]
            node2_idx = name_to_idx[edge.node2.name]
            if node1_idx < node2_idx:
                adjacency[node1_idx, node2_idx] = 1
            else:
                adjacency[node2_idx, node1_idx] = 1

    return adjacency


def run_pc_causality_lab(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    feature_names: Optional[list] = None
) -> np.ndarray:
    """
    Run PC algorithm using causality-lab implementation.

    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        feature_names: Optional list of feature names

    Returns:
        Binary adjacency matrix (n_features, n_features)
    """
    _check_availability()

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create dataset
    dataset = _create_dataset(X, feature_names)

    # Create conditional independence test
    if indep_test == 'parcorr':
        cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    elif indep_test == 'cmi':
        cond_indep_test = CondIndepCMI(dataset, threshold=alpha)
    else:
        raise ValueError(f"Unknown independence test: {indep_test}")

    # Create node set
    nodes_set = set(feature_names)

    # Run PC algorithm
    pc_learner = LearnStructPC(nodes_set, cond_indep_test)
    pc_learner.learn_structure()

    # Convert graph to adjacency matrix
    adjacency = _graph_to_adjacency(pc_learner.graph, n_features, feature_names)

    return adjacency


def run_rai(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    feature_names: Optional[list] = None
) -> np.ndarray:
    """
    Run RAI (Recursive Autonomy Identification) algorithm.

    RAI is designed for causal discovery under causal sufficiency assumption.
    It recursively identifies autonomous variables (no parents) to build the DAG.

    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        feature_names: Optional list of feature names

    Returns:
        Binary adjacency matrix (n_features, n_features)
    """
    _check_availability()

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create dataset
    dataset = _create_dataset(X, feature_names)

    # Create conditional independence test
    if indep_test == 'parcorr':
        cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    elif indep_test == 'cmi':
        cond_indep_test = CondIndepCMI(dataset, threshold=alpha)
    else:
        raise ValueError(f"Unknown independence test: {indep_test}")

    # Create node set
    nodes_set = set(feature_names)

    # Run RAI algorithm
    rai_learner = LearnStructRAI(nodes_set, cond_indep_test)
    rai_learner.learn_structure()

    # Convert graph to adjacency matrix
    adjacency = _graph_to_adjacency(rai_learner.graph, n_features, feature_names)

    return adjacency


def run_brai(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    n_bootstrap: int = 100,
    feature_names: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run B-RAI (Bootstrap/Bayesian RAI) algorithm with uncertainty estimation.

    B-RAI performs bootstrap sampling to estimate uncertainty in edge detection.
    Returns both the adjacency matrix and edge probabilities.

    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        n_bootstrap: Number of bootstrap samples
        feature_names: Optional list of feature names

    Returns:
        Tuple of:
        - Binary adjacency matrix (n_features, n_features)
        - Edge probability matrix (n_features, n_features)
    """
    _check_availability()

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create dataset
    dataset = _create_dataset(X, feature_names)

    # Create conditional independence test
    if indep_test == 'parcorr':
        cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    elif indep_test == 'cmi':
        cond_indep_test = CondIndepCMI(dataset, threshold=alpha)
    else:
        raise ValueError(f"Unknown independence test: {indep_test}")

    # Create node set
    nodes_set = set(feature_names)

    # Run B-RAI algorithm
    brai_learner = LearnStructBRAI(nodes_set, cond_indep_test, n_bootstrap=n_bootstrap)
    brai_learner.learn_structure()

    # Convert graph to adjacency matrix
    adjacency = _graph_to_adjacency(brai_learner.graph, n_features, feature_names)

    # Get edge probabilities if available
    edge_probs = np.zeros((n_features, n_features))
    if hasattr(brai_learner, 'edge_probabilities'):
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        for (source, target), prob in brai_learner.edge_probabilities.items():
            source_idx = name_to_idx[source]
            target_idx = name_to_idx[target]
            edge_probs[source_idx, target_idx] = prob

    return adjacency, edge_probs


def run_fci(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    feature_names: Optional[list] = None
) -> np.ndarray:
    """
    Run FCI (Fast Causal Inference) algorithm.

    FCI is designed for causal discovery in the presence of latent confounders.
    It returns a Partial Ancestral Graph (PAG) which we convert to adjacency.

    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        feature_names: Optional list of feature names

    Returns:
        Binary adjacency matrix (n_features, n_features)
        Note: FCI may return undirected edges due to latent confounders
    """
    _check_availability()

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create dataset
    dataset = _create_dataset(X, feature_names)

    # Create conditional independence test
    if indep_test == 'parcorr':
        cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    elif indep_test == 'cmi':
        cond_indep_test = CondIndepCMI(dataset, threshold=alpha)
    else:
        raise ValueError(f"Unknown independence test: {indep_test}")

    # Create node set
    nodes_set = set(feature_names)

    # Run FCI algorithm
    fci_learner = LearnStructFCI(nodes_set, cond_indep_test)
    fci_learner.learn_structure()

    # Convert graph to adjacency matrix
    adjacency = _graph_to_adjacency(fci_learner.graph, n_features, feature_names)

    return adjacency


def run_icd(
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    feature_names: Optional[list] = None
) -> np.ndarray:
    """
    Run ICD (Iterative Causal Discovery) algorithm.

    ICD is designed for causal discovery with latent confounders.
    It iteratively discovers causal relationships and latent structure.

    Args:
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        feature_names: Optional list of feature names

    Returns:
        Binary adjacency matrix (n_features, n_features)
    """
    _check_availability()

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Create dataset
    dataset = _create_dataset(X, feature_names)

    # Create conditional independence test
    if indep_test == 'parcorr':
        cond_indep_test = CondIndepParCorr(dataset, threshold=alpha)
    elif indep_test == 'cmi':
        cond_indep_test = CondIndepCMI(dataset, threshold=alpha)
    else:
        raise ValueError(f"Unknown independence test: {indep_test}")

    # Create node set
    nodes_set = set(feature_names)

    # Run ICD algorithm
    icd_learner = LearnStructICD(nodes_set, cond_indep_test)
    icd_learner.learn_structure()

    # Convert graph to adjacency matrix
    adjacency = _graph_to_adjacency(icd_learner.graph, n_features, feature_names)

    return adjacency


# Convenience mapping for algorithm selection
CAUSALITY_LAB_ALGORITHMS = {
    'pc': run_pc_causality_lab,
    'rai': run_rai,
    'brai': run_brai,
    'fci': run_fci,
    'icd': run_icd,
}


def run_causality_lab_algorithm(
    algorithm: str,
    X: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = 'parcorr',
    feature_names: Optional[list] = None,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Run any causality-lab algorithm by name.

    Args:
        algorithm: Algorithm name ('pc', 'rai', 'brai', 'fci', 'icd')
        X: Data matrix (n_samples, n_features)
        alpha: Significance level for independence tests
        indep_test: Independence test ('parcorr' or 'cmi')
        feature_names: Optional list of feature names
        **kwargs: Additional algorithm-specific parameters

    Returns:
        Adjacency matrix or tuple with adjacency and additional outputs
    """
    if algorithm not in CAUSALITY_LAB_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(CAUSALITY_LAB_ALGORITHMS.keys())}"
        )

    algo_func = CAUSALITY_LAB_ALGORITHMS[algorithm]
    return algo_func(X, alpha=alpha, indep_test=indep_test,
                     feature_names=feature_names, **kwargs)
