"""Information Value (IV) calculation."""

import numpy as np
import pandas as pd
from typing import Union, Dict
import warnings


class InformationValue:
    """
    Calculate Information Value (IV) for feature selection.
    
    IV is a key metric in credit scoring that measures the predictive power
    of a feature. It's based on the Weight of Evidence and quantifies how well
    a feature separates events from non-events.
    
    Formula:
    IV = Σ (% events - % non-events) × WoE
    
    Interpretation:
    - IV < 0.02: Not useful for prediction
    - 0.02 <= IV < 0.1: Weak predictive power
    - 0.1 <= IV < 0.3: Medium predictive power
    - 0.3 <= IV < 0.5: Strong predictive power
    - IV >= 0.5: Suspicious (too good to be true, check for data leakage)
    
    Parameters
    ----------
    smooth : float, default=0.5
        Smoothing parameter to avoid division by zero
    
    Examples
    --------
    >>> iv_calc = InformationValue()
    >>> iv_score = iv_calc.calculate(X_binned, y)
    >>> print(f"Information Value: {iv_score:.4f}")
    """
    
    def __init__(self, smooth: float = 0.5):
        self.smooth = smooth
        self.iv_stats_ = None
        
    def calculate(
        self,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Calculate Information Value.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        iv : float
            Information Value
        """
        X, y = self._validate_inputs(X, y)
        
        # Get unique bins
        unique_bins = np.unique(X[~pd.isna(X)])
        
        # Total counts
        total_events = np.sum(y)
        total_non_events = len(y) - total_events
        
        if total_events == 0 or total_non_events == 0:
            raise ValueError("Target must contain both events (1) and non-events (0)")
        
        iv_total = 0
        stats_list = []
        
        for bin_val in unique_bins:
            mask = X == bin_val
            bin_count = np.sum(mask)
            
            bin_events = np.sum(y[mask])
            bin_non_events = bin_count - bin_events
            
            # Apply smoothing
            bin_events_smooth = bin_events + self.smooth
            bin_non_events_smooth = bin_non_events + self.smooth
            
            # Calculate proportions
            dist_events = bin_events_smooth / (total_events + self.smooth * len(unique_bins))
            dist_non_events = bin_non_events_smooth / (total_non_events + self.smooth * len(unique_bins))
            
            # Calculate WoE
            woe = np.log(dist_events / dist_non_events)
            
            # Calculate IV for this bin
            iv_bin = (dist_events - dist_non_events) * woe
            iv_total += iv_bin
            
            stats_list.append({
                'bin': bin_val,
                'count': bin_count,
                'events': bin_events,
                'non_events': bin_non_events,
                'dist_events': dist_events,
                'dist_non_events': dist_non_events,
                'woe': woe,
                'iv': iv_bin,
            })
        
        self.iv_stats_ = pd.DataFrame(stats_list)
        self.iv_stats_['iv_cumulative'] = self.iv_stats_['iv'].cumsum()
        
        return iv_total
    
    def get_iv_stats(self) -> pd.DataFrame:
        """
        Get detailed IV statistics for each bin.
        
        Returns
        -------
        stats : DataFrame
            Detailed statistics including WoE and IV per bin
        """
        if self.iv_stats_ is None:
            raise ValueError("Must call calculate before getting statistics")
        return self.iv_stats_.copy()
    
    def interpret_iv(self, iv: float) -> str:
        """
        Interpret IV score.
        
        Parameters
        ----------
        iv : float
            Information Value
            
        Returns
        -------
        interpretation : str
            Interpretation of the IV score
        """
        if iv < 0.02:
            return "Not useful for prediction"
        elif iv < 0.1:
            return "Weak predictive power"
        elif iv < 0.3:
            return "Medium predictive power"
        elif iv < 0.5:
            return "Strong predictive power"
        else:
            return "Suspicious - very strong (check for data leakage)"
    
    def calculate_multiple(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate IV for multiple features.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Binned feature values
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        iv_summary : DataFrame
            IV scores and interpretations for all features
        """
        results = []
        
        for col in X.columns:
            try:
                iv_score = self.calculate(X[col], y)
                interpretation = self.interpret_iv(iv_score)
                
                results.append({
                    'feature': col,
                    'iv': iv_score,
                    'interpretation': interpretation,
                })
            except Exception as e:
                warnings.warn(f"Could not calculate IV for {col}: {str(e)}")
                results.append({
                    'feature': col,
                    'iv': np.nan,
                    'interpretation': 'Error',
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('iv', ascending=False).reset_index(drop=True)
        
        return df
    
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
