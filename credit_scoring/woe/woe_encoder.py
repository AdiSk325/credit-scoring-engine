"""Weight of Evidence (WoE) encoder."""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
import warnings


class WoEEncoder:
    """
    Weight of Evidence (WoE) encoder for transforming features.
    
    WoE is a powerful technique for credit scoring that:
    - Transforms categorical/binned variables to continuous
    - Captures monotonic relationship with target
    - Handles missing values naturally
    - Is interpretable in logistic regression
    
    Formula:
    WoE = ln(% of events / % of non-events)
    
    Parameters
    ----------
    smooth : float, default=0.5
        Smoothing parameter to avoid division by zero and infinite values
    handle_missing : bool, default=True
        Whether to calculate separate WoE for missing values
    min_samples : int, default=10
        Minimum samples required in a bin to calculate WoE
    
    Attributes
    ----------
    woe_mapping_ : dict
        Mapping from bin values to WoE scores
    
    Examples
    --------
    >>> encoder = WoEEncoder()
    >>> encoder.fit(X_binned, y)
    >>> X_woe = encoder.transform(X_binned)
    """
    
    def __init__(
        self,
        smooth: float = 0.5,
        handle_missing: bool = True,
        min_samples: int = 10,
    ):
        self.smooth = smooth
        self.handle_missing = handle_missing
        self.min_samples = min_samples
        self.woe_mapping_ = None
        self.woe_stats_ = None
        
    def fit(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "WoEEncoder":
        """
        Calculate WoE for each bin.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        self : WoEEncoder
            Fitted encoder
        """
        X, y = self._validate_inputs(X, y)
        
        # Calculate WoE for each bin
        self.woe_mapping_ = {}
        stats_list = []
        
        # Get unique bins
        unique_bins = np.unique(X[~pd.isna(X)])
        
        # Total counts
        total_events = np.sum(y)
        total_non_events = len(y) - total_events
        
        if total_events == 0 or total_non_events == 0:
            raise ValueError("Target must contain both events (1) and non-events (0)")
        
        for bin_val in unique_bins:
            mask = X == bin_val
            bin_count = np.sum(mask)
            
            if bin_count < self.min_samples:
                warnings.warn(
                    f"Bin {bin_val} has only {bin_count} samples (< {self.min_samples}). "
                    "WoE might be unreliable."
                )
            
            bin_events = np.sum(y[mask])
            bin_non_events = bin_count - bin_events
            
            # Apply smoothing
            bin_events_smooth = bin_events + self.smooth
            bin_non_events_smooth = bin_non_events + self.smooth
            
            # Calculate proportions
            event_rate = bin_events / bin_count if bin_count > 0 else 0
            dist_events = bin_events_smooth / (total_events + self.smooth * len(unique_bins))
            dist_non_events = bin_non_events_smooth / (total_non_events + self.smooth * len(unique_bins))
            
            # Calculate WoE
            woe = np.log(dist_events / dist_non_events)
            
            self.woe_mapping_[bin_val] = woe
            
            stats_list.append({
                'bin': bin_val,
                'count': bin_count,
                'events': bin_events,
                'non_events': bin_non_events,
                'event_rate': event_rate,
                'dist_events': dist_events,
                'dist_non_events': dist_non_events,
                'woe': woe,
            })
        
        # Handle missing values
        if self.handle_missing:
            mask_missing = pd.isna(X)
            if np.sum(mask_missing) > 0:
                bin_count = np.sum(mask_missing)
                bin_events = np.sum(y[mask_missing])
                bin_non_events = bin_count - bin_events
                
                bin_events_smooth = bin_events + self.smooth
                bin_non_events_smooth = bin_non_events + self.smooth
                
                event_rate = bin_events / bin_count if bin_count > 0 else 0
                dist_events = bin_events_smooth / (total_events + self.smooth * (len(unique_bins) + 1))
                dist_non_events = bin_non_events_smooth / (total_non_events + self.smooth * (len(unique_bins) + 1))
                
                woe_missing = np.log(dist_events / dist_non_events)
                self.woe_mapping_['__MISSING__'] = woe_missing
                
                stats_list.append({
                    'bin': '__MISSING__',
                    'count': bin_count,
                    'events': bin_events,
                    'non_events': bin_non_events,
                    'event_rate': event_rate,
                    'dist_events': dist_events,
                    'dist_non_events': dist_non_events,
                    'woe': woe_missing,
                })
        
        self.woe_stats_ = pd.DataFrame(stats_list)
        
        return self
    
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform data using WoE mapping.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Binned feature values
            
        Returns
        -------
        X_woe : array-like of shape (n_samples,)
            WoE-transformed values
        """
        if self.woe_mapping_ is None:
            raise ValueError("Must call fit before transform")
        
        X = self._validate_input(X)
        
        # Transform using WoE mapping
        X_woe = np.zeros(len(X))
        
        for i, val in enumerate(X):
            if pd.isna(val):
                if '__MISSING__' in self.woe_mapping_:
                    X_woe[i] = self.woe_mapping_['__MISSING__']
                else:
                    X_woe[i] = 0  # Default for unseen missing
            elif val in self.woe_mapping_:
                X_woe[i] = self.woe_mapping_[val]
            else:
                # Unseen bin - use median WoE
                X_woe[i] = np.median(list(self.woe_mapping_.values()))
                warnings.warn(f"Unseen bin value: {val}. Using median WoE.")
        
        return X_woe
    
    def fit_transform(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_woe_mapping(self) -> Dict:
        """Get the WoE mapping."""
        if self.woe_mapping_ is None:
            raise ValueError("Must call fit before getting mapping")
        return self.woe_mapping_.copy()
    
    def get_woe_stats(self) -> pd.DataFrame:
        """Get detailed WoE statistics."""
        if self.woe_stats_ is None:
            raise ValueError("Must call fit before getting statistics")
        return self.woe_stats_.copy()
    
    def _validate_input(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        return X
    
    def _validate_inputs(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> tuple:
        """Validate both X and y inputs."""
        X = self._validate_input(X)
        y = self._validate_input(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        # Check y is binary
        unique_y = np.unique(y[~pd.isna(y)])
        if not set(unique_y).issubset({0, 1}):
            raise ValueError("Target y must be binary (0 or 1)")
        
        return X, y
