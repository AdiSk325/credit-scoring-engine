# Credit Scoring Engine

A comprehensive Python library for building credit scoring models with a focus on:
- **Logistic regression** and its variants (L1, L2, ElasticNet regularization)
- **Variable binning** methods for continuous, categorical, and ordinal variables
- **Weight of Evidence (WoE)** and Information Value (IV) calculations
- **d-score** calculations for discrimination measurement
- **Model evaluation** metrics (Gini, KS, ROC-AUC, PSI)

## Features

### 1. Binning Module
Transform variables into bins suitable for credit scoring:

- **Continuous Variable Binning**
  - Equal-width binning
  - Equal-frequency (quantile) binning
  - Decision tree-based optimal binning
  - Custom boundaries

- **Categorical Variable Binning**
  - Rare category grouping
  - Target-based grouping
  - Custom mapping

- **Ordinal Variable Binning**
  - Preserve natural ordering
  - Group adjacent levels
  - Custom grouping

- **Binning Evaluation**
  - Monotonicity assessment
  - Chi-square independence test
  - Bin balance metrics
  - Entropy-based quality measures

### 2. Weight of Evidence (WoE) Module

- **WoE Encoder**: Transform binned variables to WoE values
- **Information Value (IV)**: Feature selection and predictive power assessment
- **d-score**: Discrimination score calculation (Cohen's d effect size)

### 3. Models Module

- **Logistic Regression Scorecard**
  - Standard logistic regression
  - L1 (Lasso), L2 (Ridge), and ElasticNet regularization
  - Feature importance and odds ratios

- **Scorecard Model**
  - Transform model to points-based scorecard
  - PDO (Points to Double Odds) calculation
  - Score to probability conversion

### 4. Metrics Module

- **Performance Metrics**
  - Gini coefficient
  - Kolmogorov-Smirnov (KS) statistic
  - ROC-AUC

- **Stability Metrics**
  - Population Stability Index (PSI)

- **Calibration Metrics**
  - Hosmer-Lemeshow test
  - Calibration curves

## Installation

```bash
# Clone the repository
git clone https://github.com/AdiSk325/credit-scoring-engine.git
cd credit-scoring-engine

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
import pandas as pd
import numpy as np
from credit_scoring.binning import ContinuousBinning, CategoricalBinning
from credit_scoring.woe import WoEEncoder, InformationValue
from credit_scoring.models import LogisticRegressionScorecard, ScorecardModel
from credit_scoring.metrics import gini_coefficient, ks_statistic, population_stability_index

# Load your data
# df = pd.read_csv('credit_data.csv')
# X = df.drop('default', axis=1)
# y = df['default']

# Example: Binning a continuous variable
binner = ContinuousBinning(method='quantile', n_bins=5)
X_binned = binner.fit_transform(X['income'], y)

# Calculate WoE
woe_encoder = WoEEncoder()
X_woe = woe_encoder.fit_transform(X_binned, y)

# Calculate Information Value
iv_calc = InformationValue()
iv_score = iv_calc.calculate(X_binned, y)
print(f"Information Value: {iv_score:.4f}")
print(f"Interpretation: {iv_calc.interpret_iv(iv_score)}")

# Train logistic regression model
model = LogisticRegressionScorecard(penalty='l2', C=1.0)
model.fit(X_woe_df, y)

# Get predictions
y_pred_proba = model.predict_proba(X_woe_test)[:, 1]

# Evaluate model
gini = gini_coefficient(y_test, y_pred_proba)
ks, ks_threshold = ks_statistic(y_test, y_pred_proba)

print(f"Gini: {gini:.4f}")
print(f"KS: {ks:.4f}")

# Create scorecard
scorecard = ScorecardModel(base_points=600, pdo=20)
scorecard.fit(model.get_coefficients(), model.get_intercept())
```

## Detailed Examples

### 1. Continuous Variable Binning

```python
from credit_scoring.binning import ContinuousBinning, BinningEvaluator

# Create binner
binner = ContinuousBinning(method='tree', max_depth=3, min_samples_leaf=50)
X_binned = binner.fit_transform(X['age'], y)

# Get bin statistics
stats = binner.get_bin_statistics(X['age'], y)
print(stats)

# Evaluate binning quality
evaluator = BinningEvaluator()
quality = evaluator.evaluate_binning_quality(X_binned, y)
print(quality)
```

### 2. Weight of Evidence and Information Value

```python
from credit_scoring.woe import WoEEncoder, InformationValue

# Calculate WoE
woe_encoder = WoEEncoder(smooth=0.5)
woe_encoder.fit(X_binned, y)

# Get WoE mapping and statistics
woe_mapping = woe_encoder.get_woe_mapping()
woe_stats = woe_encoder.get_woe_stats()
print(woe_stats)

# Transform to WoE
X_woe = woe_encoder.transform(X_binned)

# Calculate IV for multiple features
iv_calc = InformationValue()
iv_summary = iv_calc.calculate_multiple(X_binned_df, y)
print(iv_summary)
```

### 3. Building a Scorecard

```python
from credit_scoring.models import LogisticRegressionScorecard, ScorecardModel

# Train model on WoE-transformed features
model = LogisticRegressionScorecard(penalty='l2', C=1.0)
model.fit(X_woe, y)

# Get coefficients and odds ratios
coefficients = model.get_coefficients()
odds_ratios = model.get_odds_ratio()
print(coefficients)
print(odds_ratios)

# Create scorecard
scorecard = ScorecardModel(base_points=600, pdo=20, base_odds=50)
scorecard.fit(coefficients, model.get_intercept())

# Calculate points for each bin
woe_values = {
    'income': woe_encoder_income.get_woe_mapping(),
    'age': woe_encoder_age.get_woe_mapping(),
}
scorecard_points = scorecard.calculate_points(woe_values)
print(scorecard_points)

# Transform to scores
scores = scorecard.transform(X_binned_df, woe_encoders_dict)
```

### 4. Model Evaluation

```python
from credit_scoring.metrics import (
    gini_coefficient,
    ks_statistic,
    population_stability_index,
    hosmer_lemeshow_test
)

# Performance metrics
gini = gini_coefficient(y_test, y_pred_proba)
ks, threshold = ks_statistic(y_test, y_pred_proba)
print(f"Gini: {gini:.4f}, KS: {ks:.4f}")

# Stability monitoring
psi = population_stability_index(train_scores, test_scores)
print(f"PSI: {psi:.4f}")
if psi >= 0.2:
    print("Warning: Significant population shift!")

# Calibration
chi2, p_value = hosmer_lemeshow_test(y_test, y_pred_proba)
print(f"Hosmer-Lemeshow: chi2={chi2:.2f}, p-value={p_value:.4f}")
```

## Mathematical Foundations

### Weight of Evidence (WoE)

WoE measures the strength of a characteristic in separating good and bad outcomes:

```
WoE = ln(% of events / % of non-events)
```

### Information Value (IV)

IV quantifies the predictive power of a feature:

```
IV = Σ (% events - % non-events) × WoE
```

Interpretation:
- IV < 0.02: Not useful
- 0.02 ≤ IV < 0.1: Weak
- 0.1 ≤ IV < 0.3: Medium
- 0.3 ≤ IV < 0.5: Strong
- IV ≥ 0.5: Suspicious (check for data leakage)

### d-score

Standardized measure of separation:

```
d-score = (mean_events - mean_non_events) / pooled_std
```

### Scorecard Points

Convert logistic regression to points:

```
Points = (coefficient × WoE + intercept/n) × factor + offset/n
factor = PDO / ln(2)
offset = base_points - factor × ln(base_odds)
```

## Project Structure

```
credit-scoring-engine/
├── credit_scoring/
│   ├── __init__.py
│   ├── binning/
│   │   ├── __init__.py
│   │   ├── continuous.py      # Continuous variable binning
│   │   ├── categorical.py     # Categorical variable binning
│   │   ├── ordinal.py         # Ordinal variable binning
│   │   └── evaluation.py      # Binning quality evaluation
│   ├── woe/
│   │   ├── __init__.py
│   │   ├── woe_encoder.py     # WoE calculation and transformation
│   │   ├── information_value.py # IV calculation
│   │   └── dscore.py          # d-score calculation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_scorecard.py  # Logistic regression models
│   │   └── scorecard.py       # Scorecard transformation
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── performance.py     # Gini, KS, AUC
│   │   ├── stability.py       # PSI
│   │   └── calibration.py     # Hosmer-Lemeshow
│   └── utils/
│       └── __init__.py
├── tests/                     # Unit tests
├── examples/                  # Example notebooks
├── docs/                      # Documentation
├── setup.py
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## References

- Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
- Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications.