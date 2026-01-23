"""D-score calculation."""

import numpy as np
import pandas as pd
from typing import Union


class DScore:
    """
    Calculate d-score (discrimination score).
    
    The d-score is a standardized measure of separation between events and
    non-events. It's related to the Cohen's d effect size and is useful for
    assessing the discriminative power of a feature or model.
    
    Formula:
    d-score = (mean_events - mean_non_events) / pooled_std
    
    where pooled_std = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1+n2-2))
    
    Interpretation:
    - |d| < 0.2: Small effect
    - 0.2 <= |d| < 0.5: Medium effect
    - 0.5 <= |d| < 0.8: Large effect
    - |d| >= 0.8: Very large effect
    
    Examples
    --------
    >>> dscore = DScore()
    >>> d_value = dscore.calculate(X_transformed, y)
    >>> print(f"D-score: {d_value:.4f}")
    """
    
    @staticmethod
    def calculate(
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Calculate d-score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Feature values (typically WoE-transformed)
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        dscore : float
            D-score value
        """
        X, y = DScore._validate_inputs(X, y)
        
        # Split by target
        X_events = X[y == 1]
        X_non_events = X[y == 0]
        
        if len(X_events) == 0 or len(X_non_events) == 0:
            raise ValueError("Both events and non-events must be present")
        
        # Calculate means
        mean_events = np.mean(X_events)
        mean_non_events = np.mean(X_non_events)
        
        # Calculate standard deviations
        std_events = np.std(X_events, ddof=1)
        std_non_events = np.std(X_non_events, ddof=1)
        
        # Calculate pooled standard deviation
        n_events = len(X_events)
        n_non_events = len(X_non_events)
        
        pooled_std = np.sqrt(
            ((n_events - 1) * std_events**2 + (n_non_events - 1) * std_non_events**2) /
            (n_events + n_non_events - 2)
        )
        
        if pooled_std == 0:
            raise ValueError("Pooled standard deviation is zero")
        
        # Calculate d-score
        dscore = (mean_events - mean_non_events) / pooled_std
        
        return dscore
    
    @staticmethod
    def interpret_dscore(d: float) -> str:
        """
        Interpret d-score.
        
        Parameters
        ----------
        d : float
            D-score value
            
        Returns
        -------
        interpretation : str
            Interpretation of the d-score
        """
        abs_d = abs(d)
        
        if abs_d < 0.2:
            effect = "Small"
        elif abs_d < 0.5:
            effect = "Medium"
        elif abs_d < 0.8:
            effect = "Large"
        else:
            effect = "Very large"
        
        direction = "positive" if d > 0 else "negative"
        
        return f"{effect} effect ({direction} direction)"
    
    @staticmethod
    def calculate_multiple(
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate d-score for multiple features.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Feature values
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        dscore_summary : DataFrame
            D-scores and interpretations for all features
        """
        results = []
        
        for col in X.columns:
            try:
                d_value = DScore.calculate(X[col], y)
                interpretation = DScore.interpret_dscore(d_value)
                
                results.append({
                    'feature': col,
                    'dscore': d_value,
                    'abs_dscore': abs(d_value),
                    'interpretation': interpretation,
                })
            except Exception as e:
                results.append({
                    'feature': col,
                    'dscore': np.nan,
                    'abs_dscore': np.nan,
                    'interpretation': f'Error: {str(e)}',
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values('abs_dscore', ascending=False).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def calculate_with_stats(
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> dict:
        """
        Calculate d-score with detailed statistics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Feature values
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
            
        Returns
        -------
        stats : dict
            Dictionary with d-score and detailed statistics
        """
        X, y = DScore._validate_inputs(X, y)
        
        # Split by target
        X_events = X[y == 1]
        X_non_events = X[y == 0]
        
        # Calculate statistics
        mean_events = np.mean(X_events)
        mean_non_events = np.mean(X_non_events)
        std_events = np.std(X_events, ddof=1)
        std_non_events = np.std(X_non_events, ddof=1)
        
        n_events = len(X_events)
        n_non_events = len(X_non_events)
        
        pooled_std = np.sqrt(
            ((n_events - 1) * std_events**2 + (n_non_events - 1) * std_non_events**2) /
            (n_events + n_non_events - 2)
        )
        
        dscore = (mean_events - mean_non_events) / pooled_std
        
        return {
            'dscore': dscore,
            'mean_events': mean_events,
            'mean_non_events': mean_non_events,
            'std_events': std_events,
            'std_non_events': std_non_events,
            'pooled_std': pooled_std,
            'n_events': n_events,
            'n_non_events': n_non_events,
            'interpretation': DScore.interpret_dscore(dscore),
        }
    
    @staticmethod
    def _validate_input(X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        return X
    
    @staticmethod
    def _validate_inputs(
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> tuple:
        """Validate both X and y inputs."""
        X = DScore._validate_input(X)
        y = DScore._validate_input(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        # Check y is binary
        unique_y = np.unique(y[~pd.isna(y)])
        if not set(unique_y).issubset({0, 1}):
            raise ValueError("Target y must be binary (0 or 1)")
        
        return X, y
