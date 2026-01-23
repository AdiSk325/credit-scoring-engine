"""Continuous variable binning methods."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
from sklearn.tree import DecisionTreeClassifier


class ContinuousBinning:
    """
    Binning methods for continuous variables.
    
    Supports multiple binning strategies:
    - Equal width binning
    - Equal frequency (quantile) binning
    - Custom boundaries
    - Decision tree-based binning (optimal for target)
    
    Parameters
    ----------
    method : str, default='quantile'
        Binning method: 'equal_width', 'quantile', 'custom', or 'tree'
    n_bins : int, default=5
        Number of bins for equal_width and quantile methods
    custom_boundaries : list, optional
        Custom bin boundaries for 'custom' method
    max_depth : int, default=3
        Maximum depth for decision tree binning
    min_samples_leaf : int, default=50
        Minimum samples per leaf for tree binning
        
    Attributes
    ----------
    bins_ : array-like
        Fitted bin boundaries
    bin_labels_ : list
        Labels for each bin
    
    Examples
    --------
    >>> binner = ContinuousBinning(method='quantile', n_bins=5)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """
    
    def __init__(
        self,
        method: str = "quantile",
        n_bins: int = 5,
        custom_boundaries: Optional[List[float]] = None,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
    ):
        self.method = method
        self.n_bins = n_bins
        self.custom_boundaries = custom_boundaries
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bins_ = None
        self.bin_labels_ = None
        
    def fit(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "ContinuousBinning":
        """
        Fit the binning strategy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Feature values to bin
        y : array-like of shape (n_samples,), optional
            Target values (required for 'tree' method)
            
        Returns
        -------
        self : ContinuousBinning
            Fitted binner
        """
        X = self._validate_input(X)
        
        if self.method == "equal_width":
            self.bins_ = self._equal_width_bins(X)
        elif self.method == "quantile":
            self.bins_ = self._quantile_bins(X)
        elif self.method == "custom":
            if self.custom_boundaries is None:
                raise ValueError("custom_boundaries must be provided for custom method")
            self.bins_ = np.array(sorted(self.custom_boundaries))
        elif self.method == "tree":
            if y is None:
                raise ValueError("Target y is required for tree-based binning")
            y = self._validate_input(y)
            self.bins_ = self._tree_bins(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._create_bin_labels()
        return self
    
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted bins.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Feature values to bin
            
        Returns
        -------
        binned : array-like of shape (n_samples,)
            Binned values
        """
        if self.bins_ is None:
            raise ValueError("Must call fit before transform")
        
        X = self._validate_input(X)
        return np.digitize(X, self.bins_)
    
    def fit_transform(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_bin_edges(self) -> np.ndarray:
        """Get the bin edges."""
        if self.bins_ is None:
            raise ValueError("Must call fit before getting bin edges")
        return self.bins_
    
    def get_bin_labels(self) -> List[str]:
        """Get the bin labels."""
        if self.bin_labels_ is None:
            raise ValueError("Must call fit before getting bin labels")
        return self.bin_labels_
    
    def _validate_input(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        return X
    
    def _equal_width_bins(self, X: np.ndarray) -> np.ndarray:
        """Create equal-width bins."""
        min_val, max_val = np.nanmin(X), np.nanmax(X)
        return np.linspace(min_val, max_val, self.n_bins + 1)[1:-1]
    
    def _quantile_bins(self, X: np.ndarray) -> np.ndarray:
        """Create equal-frequency (quantile) bins."""
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        return np.percentile(X[~np.isnan(X)], quantiles)
    
    def _tree_bins(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create bins using decision tree."""
        # Remove NaN values
        mask = ~(np.isnan(X) | np.isnan(y))
        X_clean = X[mask].reshape(-1, 1)
        y_clean = y[mask]
        
        # Fit decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        tree.fit(X_clean, y_clean)
        
        # Extract split points
        thresholds = tree.tree_.threshold
        features = tree.tree_.feature
        
        # Get thresholds for the feature
        split_points = sorted([
            thresholds[i]
            for i in range(len(thresholds))
            if features[i] == 0 and thresholds[i] != -2
        ])
        
        return np.array(split_points)
    
    def _create_bin_labels(self):
        """Create human-readable bin labels."""
        bins_with_inf = np.concatenate([[-np.inf], self.bins_, [np.inf]])
        self.bin_labels_ = []
        
        for i in range(len(bins_with_inf) - 1):
            left = bins_with_inf[i]
            right = bins_with_inf[i + 1]
            
            if np.isinf(left):
                label = f"(-inf, {right:.2f}]"
            elif np.isinf(right):
                label = f"({left:.2f}, inf]"
            else:
                label = f"({left:.2f}, {right:.2f}]"
            
            self.bin_labels_.append(label)
    
    def get_bin_statistics(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Get statistics for each bin.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Feature values
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        stats : DataFrame
            Statistics for each bin including count, min, max, mean, and event rate if y provided
        """
        if self.bins_ is None:
            raise ValueError("Must call fit before getting statistics")
        
        X = self._validate_input(X)
        binned = self.transform(X)
        
        stats_list = []
        for i, label in enumerate(self.bin_labels_):
            mask = binned == i
            bin_data = X[mask]
            
            stat = {
                "bin": i,
                "label": label,
                "count": len(bin_data),
                "min": np.nanmin(bin_data) if len(bin_data) > 0 else np.nan,
                "max": np.nanmax(bin_data) if len(bin_data) > 0 else np.nan,
                "mean": np.nanmean(bin_data) if len(bin_data) > 0 else np.nan,
            }
            
            if y is not None:
                y_arr = self._validate_input(y)
                bin_target = y_arr[mask]
                stat["event_rate"] = np.mean(bin_target) if len(bin_target) > 0 else np.nan
                stat["event_count"] = np.sum(bin_target) if len(bin_target) > 0 else 0
            
            stats_list.append(stat)
        
        return pd.DataFrame(stats_list)
