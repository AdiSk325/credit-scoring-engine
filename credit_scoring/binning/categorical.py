"""Categorical variable binning methods."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from collections import Counter


class CategoricalBinning:
    """
    Binning methods for categorical variables.
    
    Supports multiple strategies:
    - Keep all categories separate
    - Group rare categories
    - Custom grouping
    - Target-based grouping (similar event rates)
    
    Parameters
    ----------
    method : str, default='rare'
        Binning method: 'keep_all', 'rare', 'custom', or 'target'
    rare_threshold : float, default=0.05
        Minimum frequency (as proportion) for rare category grouping
    max_categories : int, optional
        Maximum number of categories to keep
    custom_mapping : dict, optional
        Custom category mapping for 'custom' method
    
    Attributes
    ----------
    category_mapping_ : dict
        Mapping from original to binned categories
    categories_ : list
        List of final categories
    
    Examples
    --------
    >>> binner = CategoricalBinning(method='rare', rare_threshold=0.05)
    >>> binner.fit(X, y)
    >>> X_binned = binner.transform(X)
    """
    
    def __init__(
        self,
        method: str = "rare",
        rare_threshold: float = 0.05,
        max_categories: Optional[int] = None,
        custom_mapping: Optional[Dict] = None,
    ):
        self.method = method
        self.rare_threshold = rare_threshold
        self.max_categories = max_categories
        self.custom_mapping = custom_mapping
        self.category_mapping_ = None
        self.categories_ = None
        
    def fit(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "CategoricalBinning":
        """
        Fit the binning strategy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Categorical values to bin
        y : array-like of shape (n_samples,), optional
            Target values (required for 'target' method)
            
        Returns
        -------
        self : CategoricalBinning
            Fitted binner
        """
        X = self._validate_input(X)
        
        if self.method == "keep_all":
            self.category_mapping_ = self._keep_all_mapping(X)
        elif self.method == "rare":
            self.category_mapping_ = self._rare_grouping(X)
        elif self.method == "custom":
            if self.custom_mapping is None:
                raise ValueError("custom_mapping must be provided for custom method")
            self.category_mapping_ = self.custom_mapping.copy()
        elif self.method == "target":
            if y is None:
                raise ValueError("Target y is required for target-based binning")
            y = self._validate_input(y)
            self.category_mapping_ = self._target_based_grouping(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.categories_ = sorted(set(self.category_mapping_.values()))
        return self
    
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted category mapping.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Categorical values to bin
            
        Returns
        -------
        binned : array-like of shape (n_samples,)
            Binned categories
        """
        if self.category_mapping_ is None:
            raise ValueError("Must call fit before transform")
        
        X = self._validate_input(X)
        
        # Map categories, use 'OTHER' for unseen categories
        result = np.array([
            self.category_mapping_.get(x, 'OTHER')
            for x in X
        ])
        
        return result
    
    def fit_transform(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_mapping(self) -> Dict:
        """Get the category mapping."""
        if self.category_mapping_ is None:
            raise ValueError("Must call fit before getting mapping")
        return self.category_mapping_.copy()
    
    def get_categories(self) -> List:
        """Get the list of final categories."""
        if self.categories_ is None:
            raise ValueError("Must call fit before getting categories")
        return self.categories_.copy()
    
    def _validate_input(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        return X
    
    def _keep_all_mapping(self, X: np.ndarray) -> Dict:
        """Keep all categories as is."""
        unique_cats = np.unique(X[~pd.isna(X)])
        return {cat: str(cat) for cat in unique_cats}
    
    def _rare_grouping(self, X: np.ndarray) -> Dict:
        """Group rare categories together."""
        # Count frequencies
        counter = Counter(X[~pd.isna(X)])
        total = len(X[~pd.isna(X)])
        
        mapping = {}
        rare_cats = []
        
        for cat, count in counter.items():
            freq = count / total
            if freq < self.rare_threshold:
                rare_cats.append(cat)
                mapping[cat] = 'RARE'
            else:
                mapping[cat] = str(cat)
        
        # Apply max_categories if specified
        if self.max_categories is not None and len(set(mapping.values())) > self.max_categories:
            # Keep top categories by frequency
            top_cats = sorted(
                [(cat, count) for cat, count in counter.items()],
                key=lambda x: x[1],
                reverse=True
            )[:self.max_categories - 1]  # -1 for 'OTHER' category
            
            top_cat_names = {cat for cat, _ in top_cats}
            mapping = {
                cat: str(cat) if cat in top_cat_names else 'OTHER'
                for cat in counter.keys()
            }
        
        return mapping
    
    def _target_based_grouping(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Group categories with similar event rates."""
        # Calculate event rate per category
        df = pd.DataFrame({'category': X, 'target': y})
        df = df[~df['category'].isna()]
        
        event_rates = df.groupby('category')['target'].agg(['mean', 'count']).reset_index()
        event_rates.columns = ['category', 'event_rate', 'count']
        
        # Sort by event rate
        event_rates = event_rates.sort_values('event_rate')
        
        # Simple grouping: split into bins based on event rate quantiles
        n_groups = min(self.max_categories or 5, len(event_rates))
        event_rates['group'] = pd.qcut(
            event_rates['event_rate'],
            q=n_groups,
            labels=[f'GROUP_{i}' for i in range(n_groups)],
            duplicates='drop'
        )
        
        # Create mapping
        mapping = dict(zip(event_rates['category'], event_rates['group']))
        
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
            Categorical values
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        stats : DataFrame
            Statistics for each bin
        """
        if self.category_mapping_ is None:
            raise ValueError("Must call fit before getting statistics")
        
        X = self._validate_input(X)
        binned = self.transform(X)
        
        stats_list = []
        for cat in self.categories_:
            mask = binned == cat
            
            stat = {
                "category": cat,
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
