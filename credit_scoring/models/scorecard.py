"""Scorecard points calculation and transformation."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict


class ScorecardModel:
    """
    Transform logistic regression model into a scorecard with points.
    
    The scorecard converts WoE-transformed features and logistic regression
    coefficients into a points-based system commonly used in credit scoring.
    
    Formula:
    Points = (coefficient × WoE + intercept/n_features) × factor + offset/n_features
    
    where:
    - factor scales the score to a desired range
    - offset sets the base points
    
    Parameters
    ----------
    base_points : int, default=600
        Base score (points at odds of 50:50)
    pdo : int, default=20
        Points to double the odds (PDO - Points to Double Odds)
    base_odds : float, default=50
        Base odds corresponding to base_points
    
    Attributes
    ----------
    factor_ : float
        Scaling factor for points calculation
    offset_ : float
        Offset for points calculation
    scorecard_ : DataFrame
        Scorecard with points per feature
    
    Examples
    --------
    >>> scorecard = ScorecardModel(base_points=600, pdo=20)
    >>> scorecard.fit(woe_encoder, logistic_model)
    >>> scores = scorecard.transform(X_binned)
    """
    
    def __init__(
        self,
        base_points: int = 600,
        pdo: int = 20,
        base_odds: float = 50,
    ):
        self.base_points = base_points
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor_ = None
        self.offset_ = None
        self.scorecard_ = None
        self.intercept_ = None
        self.coefficients_ = None
        
    def fit(
        self,
        coefficients: Union[pd.DataFrame, Dict],
        intercept: float,
    ) -> "ScorecardModel":
        """
        Fit the scorecard model.
        
        Parameters
        ----------
        coefficients : DataFrame or dict
            Feature coefficients from logistic regression
        intercept : float
            Intercept from logistic regression
            
        Returns
        -------
        self : ScorecardModel
            Fitted scorecard
        """
        if isinstance(coefficients, pd.DataFrame):
            self.coefficients_ = coefficients.set_index('feature')['coefficient'].to_dict()
        else:
            self.coefficients_ = coefficients
        
        self.intercept_ = intercept
        
        # Calculate factor and offset
        # factor = pdo / ln(2)
        # offset = base_points - factor × ln(base_odds)
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_points - self.factor_ * np.log(self.base_odds)
        
        return self
    
    def calculate_points(
        self,
        woe_values: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Calculate scorecard points for each feature and bin.
        
        Parameters
        ----------
        woe_values : dict
            Dictionary mapping feature names to {bin: woe} dictionaries
            
        Returns
        -------
        scorecard : DataFrame
            Scorecard with points for each feature and bin
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        scorecard_list = []
        n_features = len(self.coefficients_)
        
        for feature, coef in self.coefficients_.items():
            if feature in woe_values:
                for bin_val, woe in woe_values[feature].items():
                    # Calculate points for this bin
                    # Points from feature = (coef × WoE) × factor
                    feature_points = coef * woe * self.factor_
                    
                    scorecard_list.append({
                        'feature': feature,
                        'bin': bin_val,
                        'woe': woe,
                        'coefficient': coef,
                        'points': feature_points,
                    })
        
        # Add base points (intercept contribution)
        base_contribution = self.intercept_ * self.factor_ + self.offset_
        
        self.scorecard_ = pd.DataFrame(scorecard_list)
        self.scorecard_['base_points'] = base_contribution / n_features
        self.scorecard_['total_points'] = self.scorecard_['points'] + self.scorecard_['base_points']
        
        return self.scorecard_
    
    def transform(
        self,
        X: pd.DataFrame,
        woe_encoders: Dict,
    ) -> np.ndarray:
        """
        Transform features to credit scores.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Binned features
        woe_encoders : dict
            Dictionary of WoEEncoder objects for each feature
            
        Returns
        -------
        scores : array of shape (n_samples,)
            Credit scores
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        # Transform to WoE
        X_woe = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col in woe_encoders:
                X_woe[col] = woe_encoders[col].transform(X[col])
        
        # Calculate scores
        scores = np.zeros(len(X))
        
        for feature, coef in self.coefficients_.items():
            if feature in X_woe.columns:
                scores += coef * X_woe[feature].values * self.factor_
        
        # Add base score
        scores += self.intercept_ * self.factor_ + self.offset_
        
        return scores
    
    def transform_single(
        self,
        feature_bins: Dict[str, any],
        woe_values: Dict[str, Dict],
    ) -> float:
        """
        Calculate score for a single observation.
        
        Parameters
        ----------
        feature_bins : dict
            Dictionary of feature: bin_value pairs
        woe_values : dict
            Dictionary mapping feature names to {bin: woe} dictionaries
            
        Returns
        -------
        score : float
            Credit score
        """
        if self.coefficients_ is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        score = self.intercept_ * self.factor_ + self.offset_
        
        for feature, bin_val in feature_bins.items():
            if feature in self.coefficients_ and feature in woe_values:
                if bin_val in woe_values[feature]:
                    woe = woe_values[feature][bin_val]
                    coef = self.coefficients_[feature]
                    score += coef * woe * self.factor_
        
        return score
    
    def get_scorecard(self) -> pd.DataFrame:
        """Get the complete scorecard."""
        if self.scorecard_ is None:
            raise ValueError("Scorecard not calculated. Call calculate_points first.")
        return self.scorecard_.copy()
    
    def score_to_probability(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert scorecard points to probability.
        
        Parameters
        ----------
        score : float or array
            Credit score(s)
            
        Returns
        -------
        probability : float or array
            Default probability
        """
        # Reverse the scorecard formula to get log-odds
        log_odds = (score - self.offset_) / self.factor_
        
        # Convert log-odds to probability
        odds = np.exp(log_odds)
        probability = odds / (1 + odds)
        
        return probability
    
    def probability_to_score(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert probability to scorecard points.
        
        Parameters
        ----------
        probability : float or array
            Default probability
            
        Returns
        -------
        score : float or array
            Credit score(s)
        """
        # Convert probability to odds
        odds = probability / (1 - probability)
        
        # Convert odds to log-odds
        log_odds = np.log(odds)
        
        # Convert to score
        score = self.factor_ * log_odds + self.offset_
        
        return score
