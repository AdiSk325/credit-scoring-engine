"""Calibration metrics for model evaluation."""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy import stats


def hosmer_lemeshow_test(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test.
    
    The Hosmer-Lemeshow test assesses how well the predicted probabilities
    match the observed event rates across different probability ranges.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities
    n_bins : int, default=10
        Number of bins to group predictions
        
    Returns
    -------
    chi2_statistic : float
        Chi-square test statistic
    p_value : float
        P-value (higher is better; p > 0.05 suggests good calibration)
        
    Examples
    --------
    >>> chi2, p_value = hosmer_lemeshow_test(y_true, y_pred_proba)
    >>> print(f"H-L test: chi2={chi2:.2f}, p-value={p_value:.4f}")
    >>> if p_value < 0.05:
    ...     print("Warning: Model may be poorly calibrated")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Create bins based on predicted probabilities
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
    })
    
    # Sort by predicted probability and create bins
    df = df.sort_values('y_pred')
    df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop', labels=False)
    
    # Calculate observed and expected counts per bin
    grouped = df.groupby('bin').agg({
        'y_true': ['sum', 'count'],
        'y_pred': 'mean',
    })
    
    observed_events = grouped[('y_true', 'sum')].values
    total_count = grouped[('y_true', 'count')].values
    expected_prob = grouped[('y_pred', 'mean')].values
    
    observed_non_events = total_count - observed_events
    expected_events = total_count * expected_prob
    expected_non_events = total_count * (1 - expected_prob)
    
    # Calculate chi-square statistic
    # Avoid division by zero
    chi2 = 0
    for i in range(len(observed_events)):
        if expected_events[i] > 0:
            chi2 += (observed_events[i] - expected_events[i])**2 / expected_events[i]
        if expected_non_events[i] > 0:
            chi2 += (observed_non_events[i] - expected_non_events[i])**2 / expected_non_events[i]
    
    # Degrees of freedom = number of bins - 2
    dof = len(observed_events) - 2
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    return chi2, p_value


def calibration_curve(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities
    n_bins : int, default=10
        Number of bins
        
    Returns
    -------
    prob_true : array
        True probability in each bin
    prob_pred : array
        Mean predicted probability in each bin
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Create bins
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
    })
    
    df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop', labels=False)
    
    # Calculate statistics per bin
    grouped = df.groupby('bin').agg({
        'y_true': 'mean',
        'y_pred': 'mean',
    })
    
    prob_true = grouped['y_true'].values
    prob_pred = grouped['y_pred'].values
    
    return prob_true, prob_pred
