"""Population Stability Index (PSI) for monitoring model stability."""

import numpy as np
import pandas as pd
from typing import Union
import warnings


def population_stability_index(
    expected: Union[np.ndarray, list],
    actual: Union[np.ndarray, list],
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures the shift in population distribution between two samples,
    typically used to monitor model stability over time. It compares the
    distribution of scores/features in a baseline period (expected) versus
    a new period (actual).
    
    Formula:
    PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
    
    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, investigate
    - PSI >= 0.2: Significant change, model may need recalibration
    
    Parameters
    ----------
    expected : array-like of shape (n_samples,)
        Baseline (expected) distribution
    actual : array-like of shape (n_samples,)
        New (actual) distribution to compare
    bins : int, default=10
        Number of bins for discretization
    epsilon : float, default=1e-10
        Small constant to avoid log(0)
        
    Returns
    -------
    psi : float
        Population Stability Index
        
    Examples
    --------
    >>> psi = population_stability_index(train_scores, test_scores)
    >>> print(f"PSI: {psi:.4f}")
    >>> if psi >= 0.2:
    ...     print("Warning: Significant population shift detected!")
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("Input arrays cannot be empty after removing NaN values")
    
    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)
    
    # Ensure bin edges cover both distributions
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    bin_edges[0] = min_val - epsilon
    bin_edges[-1] = max_val + epsilon
    
    # Calculate distributions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to proportions
    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)
    
    # Apply epsilon to avoid log(0)
    expected_props = np.where(expected_props == 0, epsilon, expected_props)
    actual_props = np.where(actual_props == 0, epsilon, actual_props)
    
    # Calculate PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    
    return psi


def psi_with_details(
    expected: Union[np.ndarray, list],
    actual: Union[np.ndarray, list],
    bins: int = 10,
    epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Calculate PSI with detailed breakdown by bin.
    
    Parameters
    ----------
    expected : array-like of shape (n_samples,)
        Baseline (expected) distribution
    actual : array-like of shape (n_samples,)
        New (actual) distribution to compare
    bins : int, default=10
        Number of bins for discretization
    epsilon : float, default=1e-10
        Small constant to avoid log(0)
        
    Returns
    -------
    psi_details : DataFrame
        Detailed PSI breakdown by bin
        
    Examples
    --------
    >>> details = psi_with_details(train_scores, test_scores)
    >>> print(details)
    >>> print(f"Total PSI: {details['psi'].sum():.4f}")
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    # Create bins
    _, bin_edges = np.histogram(expected, bins=bins)
    
    # Ensure bin edges cover both distributions
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    bin_edges[0] = min_val - epsilon
    bin_edges[-1] = max_val + epsilon
    
    # Calculate distributions
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to proportions
    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)
    
    # Apply epsilon to avoid log(0)
    expected_props_safe = np.where(expected_props == 0, epsilon, expected_props)
    actual_props_safe = np.where(actual_props == 0, epsilon, actual_props)
    
    # Calculate PSI per bin
    psi_per_bin = (actual_props_safe - expected_props_safe) * np.log(actual_props_safe / expected_props_safe)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'bin_lower': bin_edges[:-1],
        'bin_upper': bin_edges[1:],
        'expected_count': expected_counts,
        'actual_count': actual_counts,
        'expected_prop': expected_props,
        'actual_prop': actual_props,
        'psi': psi_per_bin,
    })
    
    return results


def interpret_psi(psi: float) -> str:
    """
    Interpret PSI value.
    
    Parameters
    ----------
    psi : float
        PSI value
        
    Returns
    -------
    interpretation : str
        Interpretation of the PSI value
    """
    if psi < 0.1:
        return "No significant change - population is stable"
    elif psi < 0.2:
        return "Moderate change - investigate further"
    else:
        return "Significant change - model may need recalibration"
