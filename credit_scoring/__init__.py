"""
Credit Scoring Engine
======================

A comprehensive Python library for building credit scoring models with focus on:
- Logistic regression and variants
- Variable binning methods (continuous, categorical, ordinal)
- Weight of Evidence (WoE) and Information Value (IV)
- d-score calculations
- Model evaluation metrics

Modules:
--------
- binning: Variable binning methods and evaluation
- woe: Weight of Evidence and Information Value calculations
- models: Logistic regression and scorecard models
- metrics: Model performance evaluation metrics
- utils: Utility functions

Example:
--------
>>> from credit_scoring.binning import ContinuousBinning
>>> from credit_scoring.woe import WoEEncoder
>>> from credit_scoring.models import ScorecardModel
"""

__version__ = "0.1.0"
__author__ = "Credit Scoring Team"

from credit_scoring.binning import (
    ContinuousBinning,
    CategoricalBinning,
    OrdinalBinning,
)
from credit_scoring.woe import WoEEncoder, InformationValue
from credit_scoring.models import ScorecardModel, LogisticRegressionScorecard
from credit_scoring.metrics import (
    gini_coefficient,
    ks_statistic,
    population_stability_index,
)

__all__ = [
    "ContinuousBinning",
    "CategoricalBinning",
    "OrdinalBinning",
    "WoEEncoder",
    "InformationValue",
    "ScorecardModel",
    "LogisticRegressionScorecard",
    "gini_coefficient",
    "ks_statistic",
    "population_stability_index",
]
