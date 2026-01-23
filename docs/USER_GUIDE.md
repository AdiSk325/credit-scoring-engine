# Credit Scoring Engine - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Guides](#detailed-guides)
5. [API Reference](#api-reference)
6. [Best Practices](#best-practices)

## Introduction

Credit Scoring Engine is a comprehensive Python library designed for building credit risk scorecards. It implements industry-standard techniques used in credit risk modeling, including variable binning, Weight of Evidence (WoE) transformation, and scorecard development.

### Key Concepts

#### Weight of Evidence (WoE)
WoE measures the strength of a characteristic in separating good and bad outcomes:
```
WoE = ln(% of events / % of non-events)
```

#### Information Value (IV)
IV quantifies the predictive power of a feature:
```
IV = Σ (% events - % non-events) × WoE
```

**Interpretation:**
- IV < 0.02: Not useful for prediction
- 0.02 ≤ IV < 0.1: Weak predictive power
- 0.1 ≤ IV < 0.3: Medium predictive power
- 0.3 ≤ IV < 0.5: Strong predictive power
- IV ≥ 0.5: Suspicious (check for data leakage)

#### d-score
Standardized measure of separation between events and non-events.

## Installation

### From Source
```bash
git clone https://github.com/AdiSk325/credit-scoring-engine.git
cd credit-scoring-engine
pip install -e .
```

### Requirements
- Python ≥ 3.8
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 1.0.0
- scipy ≥ 1.7.0

## Quick Start

```python
import pandas as pd
import numpy as np
from credit_scoring.binning import ContinuousBinning
from credit_scoring.woe import WoEEncoder, InformationValue
from credit_scoring.models import LogisticRegressionScorecard

# Sample data
X = np.random.normal(50, 10, 1000)
y = (X > 50).astype(int)

# Bin continuous variable
binner = ContinuousBinning(method='quantile', n_bins=5)
X_binned = binner.fit_transform(X, y)

# Calculate WoE
woe_encoder = WoEEncoder()
X_woe = woe_encoder.fit_transform(X_binned, y)

# Calculate Information Value
iv_calc = InformationValue()
iv = iv_calc.calculate(X_binned, y)
print(f"IV: {iv:.4f} - {iv_calc.interpret_iv(iv)}")

# Train model
model = LogisticRegressionScorecard(penalty='l2', C=1.0)
model.fit(X_woe.reshape(-1, 1), y)
```

## Detailed Guides

### 1. Variable Binning

#### Continuous Variables

```python
from credit_scoring.binning import ContinuousBinning

# Quantile binning (equal frequency)
binner = ContinuousBinning(method='quantile', n_bins=10)
X_binned = binner.fit_transform(X, y)

# Tree-based binning (optimal for target)
binner = ContinuousBinning(method='tree', max_depth=3, min_samples_leaf=50)
X_binned = binner.fit_transform(X, y)

# Custom boundaries
binner = ContinuousBinning(method='custom', custom_boundaries=[20, 40, 60, 80])
X_binned = binner.fit_transform(X)

# Get bin statistics
stats = binner.get_bin_statistics(X, y)
print(stats)
```

#### Categorical Variables

```python
from credit_scoring.binning import CategoricalBinning

# Group rare categories
binner = CategoricalBinning(method='rare', rare_threshold=0.05)
X_binned = binner.fit_transform(X_cat, y)

# Custom grouping
mapping = {'A': 'Group1', 'B': 'Group1', 'C': 'Group2'}
binner = CategoricalBinning(method='custom', custom_mapping=mapping)
X_binned = binner.fit_transform(X_cat)
```

#### Ordinal Variables

```python
from credit_scoring.binning import OrdinalBinning

# Preserve order, group adjacent levels
binner = OrdinalBinning(
    method='group_adjacent',
    n_bins=3,
    order=['low', 'medium', 'high', 'very_high']
)
X_binned = binner.fit_transform(X_ord)
```

### 2. Binning Quality Evaluation

```python
from credit_scoring.binning import BinningEvaluator

evaluator = BinningEvaluator()

# Check monotonicity
score, is_monotonic, direction = evaluator.evaluate_monotonicity(X_binned, y)
print(f"Monotonicity: {direction}, Score: {score:.2f}")

# Chi-square test
chi2, p_value = evaluator.chi_square_test(X_binned, y)
print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")

# Comprehensive evaluation
quality = evaluator.evaluate_binning_quality(X_binned, y)
print(quality)
```

### 3. WoE Transformation

```python
from credit_scoring.woe import WoEEncoder

# Create encoder
encoder = WoEEncoder(smooth=0.5, handle_missing=True)

# Fit and transform
X_woe = encoder.fit_transform(X_binned, y)

# Get WoE mapping
mapping = encoder.get_woe_mapping()
print(mapping)

# Get detailed statistics
stats = encoder.get_woe_stats()
print(stats)
```

### 4. Information Value

```python
from credit_scoring.woe import InformationValue

# Calculate IV for single feature
iv_calc = InformationValue()
iv = iv_calc.calculate(X_binned, y)
print(f"IV: {iv:.4f}")
print(f"Interpretation: {iv_calc.interpret_iv(iv)}")

# Get detailed statistics
iv_stats = iv_calc.get_iv_stats()
print(iv_stats)

# Calculate for multiple features
import pandas as pd
X_df = pd.DataFrame({'feature1': X1_binned, 'feature2': X2_binned})
iv_summary = iv_calc.calculate_multiple(X_df, y)
print(iv_summary)
```

### 5. Building Models

```python
from credit_scoring.models import LogisticRegressionScorecard

# Train model with L2 regularization
model = LogisticRegressionScorecard(penalty='l2', C=1.0)
model.fit(X_woe, y)

# Get coefficients
coefficients = model.get_coefficients()
print(coefficients)

# Get odds ratios
odds_ratios = model.get_odds_ratio()
print(odds_ratios)

# Make predictions
y_pred = model.predict_proba(X_woe_test)[:, 1]
```

### 6. Creating Scorecards

```python
from credit_scoring.models import ScorecardModel

# Create scorecard
scorecard = ScorecardModel(base_points=600, pdo=20, base_odds=50)
scorecard.fit(model.get_coefficients(), model.get_intercept())

# Calculate points for each bin
woe_values = {
    'income': woe_encoder_income.get_woe_mapping(),
    'age': woe_encoder_age.get_woe_mapping(),
}
scorecard_points = scorecard.calculate_points(woe_values)

# Transform to scores
scores = scorecard.transform(X_binned_df, woe_encoders_dict)

# Convert score to probability
prob = scorecard.score_to_probability(650)
print(f"Score 650 = {prob:.2%} default probability")
```

### 7. Model Evaluation

```python
from credit_scoring.metrics import (
    gini_coefficient,
    ks_statistic,
    population_stability_index,
    hosmer_lemeshow_test
)

# Performance metrics
gini = gini_coefficient(y_test, y_pred)
ks, threshold = ks_statistic(y_test, y_pred)
print(f"Gini: {gini:.4f}")
print(f"KS: {ks:.4f}")

# Stability
psi = population_stability_index(train_scores, test_scores)
print(f"PSI: {psi:.4f}")

# Calibration
chi2, p_value = hosmer_lemeshow_test(y_test, y_pred)
print(f"H-L test p-value: {p_value:.4f}")
```

## Best Practices

### 1. Variable Binning
- Use tree-based binning for optimal separation
- Aim for 5-10 bins for most variables
- Ensure monotonic relationship with target
- Check minimum sample size per bin (typically ≥ 5% of data)

### 2. WoE Transformation
- Always use smoothing to avoid infinite WoE values
- Handle missing values explicitly
- Check for extreme WoE values (|WoE| > 5 may indicate issues)

### 3. Feature Selection
- Use IV as initial filter (keep features with IV > 0.02)
- Check for multicollinearity among selected features
- Consider business interpretability

### 4. Model Development
- Start with L2 regularization to reduce overfitting
- Monitor train vs. test performance
- Validate scorecard logic with domain experts

### 5. Model Monitoring
- Track PSI regularly (monthly/quarterly)
- Monitor score distribution shifts
- Recalibrate when PSI > 0.2

## Common Pitfalls

1. **Too many bins**: Leads to overfitting and unstable WoE
2. **Ignoring monotonicity**: Makes scorecard less interpretable
3. **Not handling missing values**: Can lead to biased models
4. **Overfitting**: Use regularization and validate on holdout set
5. **Ignoring business constraints**: Always validate with domain experts

## Troubleshooting

### Issue: WoE values are infinite
**Solution**: Increase smoothing parameter or merge bins with zero events/non-events

### Issue: Low IV for all features
**Solution**: Check data quality, ensure target is correctly defined, try different binning strategies

### Issue: High PSI on validation set
**Solution**: May indicate data drift; consider recalibrating model or checking for data quality issues

### Issue: Poor calibration (H-L test fails)
**Solution**: Try different binning strategies, check for outliers, consider recalibration techniques

## References

1. Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
2. Anderson, R. (2007). The Credit Scoring Toolkit.
3. Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications.
