"""Ordinal variable binning methods."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict


class OrdinalBinning:
    """
    Binning methods for ordinal variables.
    
    Ordinal variables have a natural ordering but the distances between
    levels may not be equal. This class preserves the ordering while
    allowing for grouping of adjacent levels.
    
    Parameters
    ----------
    method : str, default='keep_order'
        Binning method: 'keep_order', 'group_adjacent', or 'custom'
    n_bins : int, optional
        Number of bins for 'group_adjacent' method
    custom_groups : dict, optional
        Custom grouping for 'custom' method
    order : list, optional
        Explicit ordering of categories
    
    Attributes
    ----------
    order_ : list
        Fitted ordering of categories
    group_mapping_ : dict
        Mapping from original to grouped categories
    
    Examples
    --------
    >>> binner = OrdinalBinning(method='group_adjacent', n_bins=3)
    >>> binner.fit(X, order=['low', 'medium', 'high', 'very_high'])
    >>> X_binned = binner.transform(X)
    """
    
    def __init__(
        self,
        method: str = "keep_order",
        n_bins: Optional[int] = None,
        custom_groups: Optional[Dict] = None,
        order: Optional[List] = None,
    ):
        self.method = method
        self.n_bins = n_bins
        self.custom_groups = custom_groups
        self.order = order
        self.order_ = None
        self.group_mapping_ = None
        
    def fit(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        order: Optional[List] = None,
    ) -> "OrdinalBinning":
        """
        Fit the binning strategy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Ordinal values to bin
        y : array-like of shape (n_samples,), optional
            Target values (not used currently)
        order : list, optional
            Explicit ordering of categories (overrides self.order)
            
        Returns
        -------
        self : OrdinalBinning
            Fitted binner
        """
        X = self._validate_input(X)
        
        # Determine order
        if order is not None:
            self.order_ = order
        elif self.order is not None:
            self.order_ = self.order
        else:
            # Default: use natural ordering from data
            self.order_ = sorted(np.unique(X[~pd.isna(X)]))
        
        if self.method == "keep_order":
            self.group_mapping_ = {cat: cat for cat in self.order_}
        elif self.method == "group_adjacent":
            if self.n_bins is None:
                raise ValueError("n_bins must be provided for group_adjacent method")
            self.group_mapping_ = self._group_adjacent()
        elif self.method == "custom":
            if self.custom_groups is None:
                raise ValueError("custom_groups must be provided for custom method")
            self.group_mapping_ = self.custom_groups.copy()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted grouping.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Ordinal values to bin
            
        Returns
        -------
        binned : array-like of shape (n_samples,)
            Binned categories
        """
        if self.group_mapping_ is None:
            raise ValueError("Must call fit before transform")
        
        X = self._validate_input(X)
        
        # Map categories
        result = np.array([
            self.group_mapping_.get(x, 'UNKNOWN')
            for x in X
        ])
        
        return result
    
    def fit_transform(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        order: Optional[List] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, order).transform(X)
    
    def get_mapping(self) -> Dict:
        """Get the category mapping."""
        if self.group_mapping_ is None:
            raise ValueError("Must call fit before getting mapping")
        return self.group_mapping_.copy()
    
    def get_order(self) -> List:
        """Get the ordering of categories."""
        if self.order_ is None:
            raise ValueError("Must call fit before getting order")
        return self.order_.copy()
    
    def _validate_input(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        return X
    
    def _group_adjacent(self) -> Dict:
        """Group adjacent ordinal levels."""
        n_cats = len(self.order_)
        
        if self.n_bins >= n_cats:
            # No grouping needed
            return {cat: cat for cat in self.order_}
        
        # Calculate group size
        group_size = n_cats // self.n_bins
        
        mapping = {}
        for i, cat in enumerate(self.order_):
            group_idx = min(i // group_size, self.n_bins - 1)
            # Create group label from range
            start_idx = group_idx * group_size
            end_idx = min((group_idx + 1) * group_size, n_cats) - 1
            group_label = f"{self.order_[start_idx]}_to_{self.order_[end_idx]}"
            mapping[cat] = group_label
        
        return mapping
    
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
            Ordinal values
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        stats : DataFrame
            Statistics for each bin
        """
        if self.group_mapping_ is None:
            raise ValueError("Must call fit before getting statistics")
        
        X = self._validate_input(X)
        binned = self.transform(X)
        
        unique_bins = sorted(set(self.group_mapping_.values()))
        
        stats_list = []
        for bin_label in unique_bins:
            mask = binned == bin_label
            
            stat = {
                "bin": bin_label,
                "count": np.sum(mask),
                "proportion": np.mean(mask),
            }
            
            if y is not None:
                y_arr = self._validate_input(y)
                bin_target = y_arr[mask]
                stat["event_rate"] = np.mean(bin_target) if len(bin_target) > 0 else np.nan
                stat["event_count"] = np.sum(bin_target) if len(bin_target) > 0 else 0
            
            stats_list.append(stat)
        
        return pd.DataFrame(stats_list)
