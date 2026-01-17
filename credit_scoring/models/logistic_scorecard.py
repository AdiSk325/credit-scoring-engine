"""Logistic regression-based scorecard model."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionScorecard:
    """
    Logistic regression model with scorecard transformation.
    
    This class wraps scikit-learn's LogisticRegression and adds:
    - Support for regularization (L1, L2, ElasticNet)
    - Scorecard points calculation
    - Feature importance based on WoE coefficients
    
    Parameters
    ----------
    penalty : str, default='l2'
        Regularization penalty: 'l1', 'l2', 'elasticnet', or 'none'
    C : float, default=1.0
        Inverse of regularization strength (smaller = stronger)
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only used if penalty='elasticnet')
    max_iter : int, default=1000
        Maximum iterations for solver
    random_state : int, optional
        Random state for reproducibility
    
    Attributes
    ----------
    model_ : LogisticRegression
        Fitted logistic regression model
    feature_names_ : list
        Names of features used in training
    
    Examples
    --------
    >>> model = LogisticRegressionScorecard(penalty='l2', C=1.0)
    >>> model.fit(X_woe, y)
    >>> predictions = model.predict_proba(X_woe_test)
    """
    
    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None
        self.feature_names_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[list] = None,
    ) -> "LogisticRegressionScorecard":
        """
        Fit the logistic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features (typically WoE-transformed)
        y : array-like of shape (n_samples,)
            Binary target (0 or 1)
        feature_names : list, optional
            Names of features (if X is numpy array)
            
        Returns
        -------
        self : LogisticRegressionScorecard
            Fitted model
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Configure solver based on penalty
        if self.penalty == 'elasticnet':
            solver = 'saga'
        elif self.penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = 'lbfgs'
        
        # Create and fit model
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            l1_ratio=self.l1_ratio if self.penalty == 'elasticnet' else None,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        
        self.model_.fit(X, y)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict
            
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict
            
        Returns
        -------
        probabilities : array of shape (n_samples, 2)
            Predicted probabilities for each class
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict_proba(X)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients.
        
        Returns
        -------
        coefficients : DataFrame
            Feature names and their coefficients
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.model_.coef_[0],
        })
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        return self.model_.intercept_[0]
    
    def get_odds_ratio(self) -> pd.DataFrame:
        """
        Get odds ratios for each feature.
        
        Returns
        -------
        odds_ratios : DataFrame
            Feature names and their odds ratios
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.model_.coef_[0],
            'odds_ratio': np.exp(self.model_.coef_[0]),
        })
        
        return coef_df
