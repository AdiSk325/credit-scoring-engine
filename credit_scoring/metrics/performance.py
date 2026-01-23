"""Performance metrics for credit scoring models."""

import numpy as np
from typing import Union, Tuple
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score, roc_curve


def gini_coefficient(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    Calculate Gini coefficient.
    
    The Gini coefficient is a common metric in credit scoring that
    measures the discriminatory power of a model. It's related to
    the area under the ROC curve (AUC):
    
    Gini = 2 × AUC - 1
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores
        
    Returns
    -------
    gini : float
        Gini coefficient (range: -1 to 1, higher is better)
        
    Examples
    --------
    >>> gini = gini_coefficient(y_true, y_pred_proba)
    >>> print(f"Gini: {gini:.4f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    auc = sklearn_roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    
    return gini


def ks_statistic(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov (KS) statistic.
    
    The KS statistic measures the maximum separation between the
    cumulative distributions of events and non-events. It's a
    popular metric in credit scoring.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores
        
    Returns
    -------
    ks : float
        KS statistic (range: 0 to 1, higher is better)
    ks_threshold : float
        Threshold at which KS statistic is maximized
        
    Examples
    --------
    >>> ks, threshold = ks_statistic(y_true, y_pred_proba)
    >>> print(f"KS: {ks:.4f} at threshold {threshold:.4f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # KS is the maximum difference between TPR and FPR
    ks = np.max(tpr - fpr)
    
    # Find the threshold at which KS is maximized
    ks_idx = np.argmax(tpr - fpr)
    ks_threshold = thresholds[ks_idx]
    
    return ks, ks_threshold


def roc_auc_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Wrapper around sklearn's roc_auc_score for consistency.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores
        
    Returns
    -------
    auc : float
        AUC score (range: 0 to 1, higher is better)
        
    Examples
    --------
    >>> auc = roc_auc_score(y_true, y_pred_proba)
    >>> print(f"AUC: {auc:.4f}")
    """
    return sklearn_roc_auc_score(y_true, y_pred)


def calculate_performance_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores
        
    Returns
    -------
    metrics : dict
        Dictionary containing all performance metrics
        
    Examples
    --------
    >>> metrics = calculate_performance_metrics(y_true, y_pred_proba)
    >>> for name, value in metrics.items():
    ...     print(f"{name}: {value:.4f}")
    """
    auc = roc_auc_score(y_true, y_pred)
    gini = gini_coefficient(y_true, y_pred)
    ks, ks_threshold = ks_statistic(y_true, y_pred)
    
    return {
        'auc': auc,
        'gini': gini,
        'ks': ks,
        'ks_threshold': ks_threshold,
    }
