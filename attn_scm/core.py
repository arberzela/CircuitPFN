"""
Core Attn-SCM implementation.

This module provides the main AttnSCM class that orchestrates the complete
pipeline for extracting causal graphs from TabPFN attention maps.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings

# Try to import TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    warnings.warn("TabPFN not installed. Install with: pip install tabpfn")

from attn_scm.attention import AttentionExtractor, aggregate_attention_across_batch
from attn_scm.heads import identify_structural_heads, compute_head_scores
from attn_scm.graph import (
    aggregate_heads_to_adjacency,
    threshold_adjacency,
    enforce_directionality,
    extract_markov_blanket,
    validate_dag
)


class AttnSCM:
    """
    Zero-shot causal graph extraction from TabPFN attention maps.
    
    This class implements the complete Attn-SCM pipeline:
    1. Extract attention maps from pre-trained TabPFN
    2. Identify structural heads via entropy filtering
    3. Aggregate into raw adjacency matrix
    4. Apply post-processing (thresholding, directionality)
    
    Example:
        >>> model = AttnSCM(top_k_heads=5)
        >>> adjacency = model.fit(X, y)
        >>> mb = model.get_markov_blanket(target_idx=0)
    """
    
    def __init__(
        self,
        top_k_heads: int = 5,
        threshold_method: str = 'otsu',
        threshold_value: Optional[float] = None,
        directionality_method: str = 'asymmetry',
        head_scoring_method: str = 'entropy',
        layer_range: Optional[Tuple[int, int]] = None,
        device: str = 'cpu',
        aggregate_batch: str = 'mean',
        aggregate_heads: str = 'mean'
    ):
        """
        Initialize Attn-SCM.
        
        Args:
            top_k_heads: Number of structural heads to select
            threshold_method: Method for sparsifying adjacency ('otsu', 'fixed', 'top_k')
            threshold_value: Threshold value for 'fixed' method
            directionality_method: Method for enforcing directed edges ('asymmetry', 'dag')
            head_scoring_method: Method for scoring heads ('entropy', 'sparsity', 'combined')
            layer_range: Optional (min_layer, max_layer) to restrict structural head search
            device: Device for TabPFN ('cpu' or 'cuda')
            aggregate_batch: How to aggregate attention across batches
            aggregate_heads: How to aggregate attention across heads
        """
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is required. Install with: pip install tabpfn")
        
        self.top_k_heads = top_k_heads
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.directionality_method = directionality_method
        self.head_scoring_method = head_scoring_method
        self.layer_range = layer_range
        self.device = device
        self.aggregate_batch = aggregate_batch
        self.aggregate_heads = aggregate_heads
        
        # Initialize TabPFN model
        self.tabpfn_model = TabPFNClassifier(device=device)
        
        # Storage for extracted information
        self.attention_dict_ = None
        self.structural_heads_ = None
        self.adjacency_raw_ = None
        self.adjacency_ = None
        self.feature_names_ = None
        self.n_features_ = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract causal graph from data.
        
        Args:
            X: Input features (N, d)
            y: Target labels (N,)
            feature_names: Optional names for features
            
        Returns:
            Final adjacency matrix (d, d)
        """
        # Validate input
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {len(y)}")
        
        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names or [f"X{i}" for i in range(self.n_features_)]
        
        # Step 1: Extract attention maps
        print(f"Extracting attention maps from TabPFN...")
        extractor = AttentionExtractor(self.tabpfn_model)
        
        # Note: TabPFN has constraints (max 1000 samples, 100 features)
        if X.shape[0] > 1000:
            warnings.warn(f"TabPFN supports max 1000 samples. Using first 1000.")
            X = X[:1000]
            y = y[:1000]
        if X.shape[1] > 100:
            warnings.warn(f"TabPFN supports max 100 features. Using first 100.")
            X = X[:, :100]
            self.n_features_ = 100
        
        self.attention_dict_ = extractor.extract_attention(X, y)
        
        # Aggregate across batch dimension
        for layer_idx in self.attention_dict_:
            attn = self.attention_dict_[layer_idx]
            if attn.ndim == 4:  # Has batch dimension
                self.attention_dict_[layer_idx] = aggregate_attention_across_batch(
                    attn, aggregation=self.aggregate_batch
                )
        
        # Step 2: Identify structural heads
        print(f"Identifying top-{self.top_k_heads} structural heads...")
        self.structural_heads_ = identify_structural_heads(
            self.attention_dict_,
            top_k=self.top_k_heads,
            method=self.head_scoring_method,
            layer_range=self.layer_range
        )
        
        print(f"Selected heads: {self.structural_heads_}")
        
        # Step 3: Aggregate to adjacency matrix
        print(f"Aggregating attention into adjacency matrix...")
        self.adjacency_raw_ = aggregate_heads_to_adjacency(
            self.attention_dict_,
            self.structural_heads_,
            aggregation=self.aggregate_heads
        )
        
        # Step 4: Threshold for sparsity
        print(f"Applying {self.threshold_method} thresholding...")
        adjacency_thresh = threshold_adjacency(
            self.adjacency_raw_,
            method=self.threshold_method,
            threshold=self.threshold_value
        )
        
        # Step 5: Enforce directionality
        print(f"Enforcing directionality via {self.directionality_method}...")
        self.adjacency_ = enforce_directionality(
            adjacency_thresh,
            method=self.directionality_method
        )
        
        # Validate
        validation = validate_dag(self.adjacency_)
        print(f"Final graph: {validation['num_edges']} edges, "
              f"sparsity={validation['sparsity']:.2f}, "
              f"is_DAG={validation['is_dag']}")
        
        return self.adjacency_
    
    def get_adjacency_matrix(self, binarize: bool = False) -> np.ndarray:
        """
        Get the extracted adjacency matrix.
        
        Args:
            binarize: Whether to return binary adjacency (0/1)
            
        Returns:
            Adjacency matrix (d, d)
        """
        if self.adjacency_ is None:
            raise ValueError("Must call fit() first")
        
        if binarize:
            return (self.adjacency_ > 0).astype(int)
        return self.adjacency_
    
    def get_markov_blanket(self, target_idx: int) -> List[int]:
        """
        Get the Markov Blanket of a target variable.
        
        Args:
            target_idx: Index of target variable
            
        Returns:
            List of feature indices in the Markov Blanket
        """
        if self.adjacency_ is None:
            raise ValueError("Must call fit() first")
        
        return extract_markov_blanket(self.adjacency_, target_idx)
    
    def get_parents(self, node_idx: int) -> List[int]:
        """
        Get parent nodes (incoming edges) of a given node.
        
        Args:
            node_idx: Index of node
            
        Returns:
            List of parent node indices
        """
        if self.adjacency_ is None:
            raise ValueError("Must call fit() first")
        
        binary_adj = (self.adjacency_ > 0).astype(int)
        parents = np.where(binary_adj[:, node_idx] > 0)[0]
        return parents.tolist()
    
    def get_children(self, node_idx: int) -> List[int]:
        """
        Get child nodes (outgoing edges) of a given node.
        
        Args:
            node_idx: Index of node
            
        Returns:
            List of child node indices
        """
        if self.adjacency_ is None:
            raise ValueError("Must call fit() first")
        
        binary_adj = (self.adjacency_ > 0).astype(int)
        children = np.where(binary_adj[node_idx, :] > 0)[0]
        return children.tolist()
    
    def extract_for_layer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer_idx: int,
        head_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract adjacency matrix using attention from a specific layer only.
        
        This is useful for Experiment B (layer-wise localization).
        
        Args:
            X: Input features (N, d)
            y: Target labels (N,)
            layer_idx: Layer to extract from
            head_indices: Specific heads to use (None = all heads in layer)
            
        Returns:
            Adjacency matrix from this layer (d, d)
        """
        # Extract attention
        extractor = AttentionExtractor(self.tabpfn_model)
        attention_dict = extractor.extract_attention(X, y)
        
        if layer_idx not in attention_dict:
            raise ValueError(f"Layer {layer_idx} not found in attention dict")
        
        # Get attention for this layer
        layer_attn = attention_dict[layer_idx]
        
        # Aggregate across batch
        if layer_attn.ndim == 4:
            layer_attn = aggregate_attention_across_batch(layer_attn)
        
        # Select heads
        if head_indices is None:
            n_heads = layer_attn.shape[0]
            head_indices = list(range(n_heads))
        
        # Create structural_heads format
        structural_heads = [(layer_idx, h) for h in head_indices]
        
        # Aggregate
        adjacency_raw = aggregate_heads_to_adjacency(
            {layer_idx: layer_attn},
            structural_heads,
            aggregation=self.aggregate_heads
        )
        
        # Post-process
        adjacency_thresh = threshold_adjacency(
            adjacency_raw,
            method=self.threshold_method,
            threshold=self.threshold_value
        )
        
        adjacency = enforce_directionality(
            adjacency_thresh,
            method=self.directionality_method
        )
        
        return adjacency
