"""Adapter for using TabPFN predictions with causality-lab algorithms.

This module provides custom conditional independence tests that leverage
TabPFN's predictions and attention patterns to inform causal discovery.
"""

import numpy as np
from typing import Optional, Set, Tuple
from scipy import stats

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

try:
    from causality_lab.cond_indep_tests import CondIndepTest
    from causality_lab.data import Dataset
    CAUSALITY_LAB_AVAILABLE = True
except ImportError:
    CAUSALITY_LAB_AVAILABLE = False


class CondIndepTabPFN(CondIndepTest if CAUSALITY_LAB_AVAILABLE else object):
    """
    Conditional independence test using TabPFN predictions.

    This test uses TabPFN's ability to capture complex patterns in tabular data
    to assess conditional independence. It trains TabPFN to predict one variable
    from others and uses prediction accuracy as a proxy for dependence.

    The intuition: If X is independent of Y given Z, then adding Y to the
    conditioning set Z shouldn't improve predictions of X.
    """

    def __init__(
        self,
        dataset: 'Dataset',
        threshold: float = 0.05,
        device: str = 'cpu',
        prediction_threshold: float = 0.01
    ):
        """
        Initialize TabPFN-based conditional independence test.

        Args:
            dataset: Dataset object containing the data
            threshold: Significance level for independence test
            device: Device for TabPFN ('cpu' or 'cuda')
            prediction_threshold: Minimum improvement in prediction to declare dependence
        """
        if not TABPFN_AVAILABLE:
            raise ImportError("tabpfn is required. Install with: pip install tabpfn")
        if not CAUSALITY_LAB_AVAILABLE:
            raise ImportError("causality-lab is required")

        super().__init__(dataset, threshold)
        self.device = device
        self.prediction_threshold = prediction_threshold

    def test(self, x: str, y: str, z: Set[str]) -> Tuple[bool, float]:
        """
        Test if X is independent of Y given Z using TabPFN.

        Args:
            x: Target variable name
            y: Variable to test independence with
            z: Conditioning set (variable names)

        Returns:
            Tuple of (is_independent, p_value)
        """
        # Get data
        X_data = self.dataset.data
        var_names = self.dataset.var_names
        name_to_idx = {name: idx for idx, name in enumerate(var_names)}

        x_idx = name_to_idx[x]
        y_idx = name_to_idx[y]
        z_indices = [name_to_idx[z_var] for z_var in z] if z else []

        # Discretize target for classification
        x_target = X_data[:, x_idx]
        x_target_discrete = self._discretize(x_target)

        # Test 1: Predict X from Z only
        if len(z_indices) > 0:
            X_z = X_data[:, z_indices]
            score_z = self._predict_with_tabpfn(X_z, x_target_discrete)
        else:
            # No conditioning set: just use prior (random guess)
            score_z = 1.0 / len(np.unique(x_target_discrete))

        # Test 2: Predict X from Z ∪ {Y}
        zy_indices = z_indices + [y_idx]
        X_zy = X_data[:, zy_indices]
        score_zy = self._predict_with_tabpfn(X_zy, x_target_discrete)

        # Compute improvement
        improvement = score_zy - score_z

        # If improvement is negligible, declare independence
        is_independent = improvement < self.prediction_threshold

        # Convert improvement to pseudo p-value (0 to 1)
        # Higher improvement -> lower p-value (more dependent)
        p_value = max(0.0, min(1.0, 1.0 - improvement))

        return is_independent, p_value

    def _discretize(self, x: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """
        Discretize continuous variable for classification.

        Args:
            x: Continuous variable
            n_bins: Number of bins

        Returns:
            Discretized labels
        """
        # Use quantile-based binning
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(x, quantiles)
        bin_edges[-1] += 1e-10  # Ensure last value is included
        labels = np.digitize(x, bin_edges[1:-1])
        return labels

    def _predict_with_tabpfn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3
    ) -> float:
        """
        Train TabPFN and return prediction accuracy.

        Args:
            X: Features
            y: Target labels
            test_size: Fraction of data for testing

        Returns:
            Prediction accuracy (0 to 1)
        """
        # Handle edge cases
        if X.shape[0] < 10:
            return 0.5  # Not enough data

        if X.shape[1] == 0:
            # No features: return prior accuracy
            unique, counts = np.unique(y, return_counts=True)
            return np.max(counts) / len(y)

        # Enforce TabPFN limits
        n_samples = min(X.shape[0], 1000)
        n_features = min(X.shape[1], 100)
        X = X[:n_samples, :n_features]
        y = y[:n_samples]

        # Split data
        n_train = int(n_samples * (1 - test_size))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        if len(X_test) < 1:
            return 0.5

        try:
            # Train TabPFN
            model = TabPFNClassifier(device=self.device)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Compute accuracy
            accuracy = np.mean(y_pred == y_test)
            return accuracy

        except Exception as e:
            # If TabPFN fails, fall back to baseline
            unique, counts = np.unique(y_train, return_counts=True)
            return np.max(counts) / len(y_train)


class CondIndepAttentionWeighted(CondIndepTest if CAUSALITY_LAB_AVAILABLE else object):
    """
    Conditional independence test weighted by TabPFN attention patterns.

    This test combines standard partial correlation with attention weights
    from TabPFN to prioritize relationships the model considers important.
    """

    def __init__(
        self,
        dataset: 'Dataset',
        threshold: float = 0.05,
        attention_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize attention-weighted conditional independence test.

        Args:
            dataset: Dataset object containing the data
            threshold: Significance level for independence test
            attention_weights: Precomputed attention matrix (n_features, n_features)
        """
        if not CAUSALITY_LAB_AVAILABLE:
            raise ImportError("causality-lab is required")

        super().__init__(dataset, threshold)
        self.attention_weights = attention_weights

    def test(self, x: str, y: str, z: Set[str]) -> Tuple[bool, float]:
        """
        Test if X is independent of Y given Z using attention-weighted correlation.

        Args:
            x: Target variable name
            y: Variable to test independence with
            z: Conditioning set (variable names)

        Returns:
            Tuple of (is_independent, p_value)
        """
        # Get data
        X_data = self.dataset.data
        var_names = self.dataset.var_names
        name_to_idx = {name: idx for idx, name in enumerate(var_names)}

        x_idx = name_to_idx[x]
        y_idx = name_to_idx[y]
        z_indices = [name_to_idx[z_var] for z_var in z] if z else []

        # Compute partial correlation
        if len(z_indices) == 0:
            # No conditioning: use simple correlation
            corr = np.corrcoef(X_data[:, x_idx], X_data[:, y_idx])[0, 1]
            n = len(X_data)
            # Fisher's z-transformation for p-value
            if abs(corr) >= 1.0:
                p_value = 0.0
            else:
                z_stat = 0.5 * np.log((1 + corr) / (1 - corr))
                z_stat_normalized = z_stat * np.sqrt(n - 3)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat_normalized)))
        else:
            # Partial correlation given Z
            corr, p_value = self._partial_correlation(
                X_data[:, x_idx],
                X_data[:, y_idx],
                X_data[:, z_indices]
            )

        # Weight by attention if available
        if self.attention_weights is not None:
            attention_weight = self.attention_weights[x_idx, y_idx]
            # Modify p-value: stronger attention -> more likely dependent
            p_value = p_value * (1 - attention_weight * 0.5)

        is_independent = p_value > self.threshold
        return is_independent, p_value

    def _partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute partial correlation between x and y given z.

        Args:
            x: First variable
            y: Second variable
            z: Conditioning variables (n_samples, n_conditioning)

        Returns:
            Tuple of (partial_correlation, p_value)
        """
        n = len(x)

        # Regress x on z
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        from sklearn.linear_model import LinearRegression

        reg_x = LinearRegression().fit(z, x)
        res_x = x - reg_x.predict(z)

        reg_y = LinearRegression().fit(z, y)
        res_y = y - reg_y.predict(z)

        # Correlation of residuals
        corr = np.corrcoef(res_x, res_y)[0, 1]

        # Fisher's z-transformation
        if abs(corr) >= 1.0:
            p_value = 0.0
        else:
            z_stat = 0.5 * np.log((1 + corr) / (1 - corr))
            # Degrees of freedom: n - |z| - 3
            df = n - z.shape[1] - 3
            z_stat_normalized = z_stat * np.sqrt(max(df, 1))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat_normalized)))

        return corr, p_value
