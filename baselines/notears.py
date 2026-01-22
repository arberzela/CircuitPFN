"""NOTEARS algorithm baseline."""

import numpy as np
from typing import Optional
from scipy.optimize import minimize
from scipy.special import expit


def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    loss_type: str = 'l2',
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e+16,
    w_threshold: float = 0.3
) -> np.ndarray:
    """
    NOTEARS algorithm for linear causal discovery.
    
    Solves: min_W l(W) + lambda1 * ||W||_1 s.t. h(W) = 0
    where h(W) = tr(e^{W ⊙ W}) - d = 0 enforces acyclicity
    
    Args:
        X: Data matrix (n_samples, n_features)
        lambda1: L1 regularization parameter
        loss_type: Loss function ('l2' or 'logistic')
        max_iter: Maximum number of iterations
        h_tol: Tolerance for acyclicity constraint
        rho_max: Maximum value of penalty parameter
        w_threshold: Threshold for edge pruning
        
    Returns:
        Adjacency matrix (n_features, n_features)
        
    Reference:
        Zheng et al. "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
        NeurIPS 2018
    """
    n, d = X.shape
    
    def _loss(W):
        """Compute loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
        elif loss_type == 'logistic':
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')
        return loss
    
    def _h(W):
        """Compute acyclicity constraint."""
        E = np.exp(W * W)  # Element-wise
        h = np.trace(E) - d
        return h
    
    def _adj(w):
        """Convert from vector to matrix."""
        return w.reshape([d, d])
    
    def _func(w):
        """Compute objective for optimization."""
        W = _adj(w)
        loss = _loss(W)
        h_val = _h(W)
        return loss + lambda1 * np.abs(W).sum()
    
    # Initialize
    w_est = np.zeros(d * d)
    
    # NOTEARS uses augmented Lagrangian method
    rho, alpha, h_val = 1.0, 0.0, np.inf
    
    for iteration in range(max_iter):
        # Define augmented Lagrangian
        def _aug_lagrangian(w):
            W = _adj(w)
            loss = _loss(W)
            h_val = _h(W)
            l1_penalty = lambda1 * np.abs(W).sum()
            return loss + l1_penalty + alpha * h_val + 0.5 * rho * h_val * h_val
        
        # Optimize with L-BFGS-B
        res = minimize(
            _aug_lagrangian,
            w_est,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        w_est = res.x
        
        # Check acyclicity constraint
        h_val = _h(_adj(w_est))
        
        if h_val > 0.25 * h_tol:
            rho *= 10
        else:
            break
        
        if rho >= rho_max:
            break
        
        # Update Lagrange multiplier
        alpha += rho * h_val
    
    # Threshold
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    
    return W_est


def run_notears(
    X: np.ndarray,
    lambda1: float = 0.1,
    w_threshold: float = 0.3
) -> np.ndarray:
    """
    Wrapper function to run NOTEARS algorithm.
    
    Args:
        X: Data matrix (n_samples, n_features)
        lambda1: L1 regularization parameter
        w_threshold: Threshold for edge pruning
        
    Returns:
        Binary adjacency matrix (n_features, n_features)
    """
    W_est = notears_linear(X, lambda1=lambda1, w_threshold=w_threshold)
    
    # Binarize
    adjacency = (np.abs(W_est) > 0).astype(int)
    
    return adjacency
