"""Binning evaluation methods."""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy import stats


class BinningEvaluator:
    """
    Evaluate the quality and effectiveness of binning.
    
    Provides various metrics to assess binning quality:
    - Monotonicity of event rates
    - Chi-square test for independence
    - Entropy-based measures
    - Bin balance metrics
    
    Examples
    --------
    >>> evaluator = BinningEvaluator()
    >>> score = evaluator.evaluate_monotonicity(binned_X, y)
    >>> chi2, p_value = evaluator.chi_square_test(binned_X, y)
    """
    
    @staticmethod
    def evaluate_monotonicity(
        binned_X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[float, bool, str]:
        """
        Evaluate monotonicity of event rates across bins.
        
        A good binning should show monotonic relationship with target.
        
        Parameters
        ----------
        binned_X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Target values (binary)
            
        Returns
        -------
        score : float
            Monotonicity score (0 to 1, higher is better)
        is_monotonic : bool
            Whether the relationship is monotonic
        direction : str
            'increasing', 'decreasing', or 'non-monotonic'
        """
        # Calculate event rate per bin
        df = pd.DataFrame({'bin': binned_X, 'target': y})
        bin_stats = df.groupby('bin')['target'].agg(['mean', 'count']).reset_index()
        bin_stats = bin_stats.sort_values('bin')
        
        event_rates = bin_stats['mean'].values
        
        # Check monotonicity
        is_increasing = all(event_rates[i] <= event_rates[i+1] 
                           for i in range(len(event_rates)-1))
        is_decreasing = all(event_rates[i] >= event_rates[i+1] 
                           for i in range(len(event_rates)-1))
        
        if is_increasing:
            direction = 'increasing'
            is_monotonic = True
            score = 1.0
        elif is_decreasing:
            direction = 'decreasing'
            is_monotonic = True
            score = 1.0
        else:
            direction = 'non-monotonic'
            is_monotonic = False
            # Calculate violation score
            n_violations = 0
            for i in range(len(event_rates) - 1):
                if event_rates[i] > event_rates[i+1]:
                    n_violations += 1
            score = 1.0 - (n_violations / (len(event_rates) - 1))
        
        return score, is_monotonic, direction
    
    @staticmethod
    def chi_square_test(
        binned_X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Tuple[float, float]:
        """
        Perform chi-square test for independence.
        
        Tests whether the binned feature and target are independent.
        Lower p-value indicates stronger relationship.
        
        Parameters
        ----------
        binned_X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Target values (binary)
            
        Returns
        -------
        chi2_statistic : float
            Chi-square test statistic
        p_value : float
            P-value of the test
        """
        # Create contingency table
        df = pd.DataFrame({'bin': binned_X, 'target': y})
        contingency_table = pd.crosstab(df['bin'], df['target'])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return chi2, p_value
    
    @staticmethod
    def calculate_bin_balance(
        binned_X: Union[pd.Series, np.ndarray],
    ) -> Tuple[float, float]:
        """
        Calculate bin balance metrics.
        
        Parameters
        ----------
        binned_X : array-like of shape (n_samples,)
            Binned feature values
            
        Returns
        -------
        gini_index : float
            Gini index of bin distribution (0 = perfect balance, 1 = complete imbalance)
        min_proportion : float
            Proportion of samples in smallest bin
        """
        # Count samples per bin
        bin_counts = pd.Series(binned_X).value_counts(normalize=True).values
        
        # Calculate Gini index
        n_bins = len(bin_counts)
        sorted_counts = np.sort(bin_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n_bins + 1)) * sorted_counts) / 
                (n_bins * np.sum(sorted_counts))) - (n_bins + 1) / n_bins
        
        min_proportion = np.min(bin_counts)
        
        return gini, min_proportion
    
    @staticmethod
    def calculate_entropy(
        binned_X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Calculate entropy of target within bins.
        
        Lower entropy indicates better separation.
        
        Parameters
        ----------
        binned_X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Target values (binary)
            
        Returns
        -------
        entropy : float
            Weighted average entropy across bins
        """
        df = pd.DataFrame({'bin': binned_X, 'target': y})
        
        total_entropy = 0
        total_count = len(df)
        
        for bin_val in df['bin'].unique():
            bin_data = df[df['bin'] == bin_val]['target']
            bin_count = len(bin_data)
            
            # Calculate entropy for this bin
            if bin_count > 0:
                p1 = np.mean(bin_data)
                p0 = 1 - p1
                
                if p1 > 0 and p0 > 0:
                    bin_entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))
                else:
                    bin_entropy = 0
                
                total_entropy += (bin_count / total_count) * bin_entropy
        
        return total_entropy
    
    @staticmethod
    def evaluate_binning_quality(
        binned_X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Comprehensive evaluation of binning quality.
        
        Parameters
        ----------
        binned_X : array-like of shape (n_samples,)
            Binned feature values
        y : array-like of shape (n_samples,)
            Target values (binary)
            
        Returns
        -------
        results : DataFrame
            Comprehensive quality metrics
        """
        # Monotonicity
        mono_score, is_mono, direction = BinningEvaluator.evaluate_monotonicity(binned_X, y)
        
        # Chi-square test
        chi2, p_value = BinningEvaluator.chi_square_test(binned_X, y)
        
        # Balance
        gini, min_prop = BinningEvaluator.calculate_bin_balance(binned_X)
        
        # Entropy
        entropy = BinningEvaluator.calculate_entropy(binned_X, y)
        
        results = pd.DataFrame({
            'metric': [
                'monotonicity_score',
                'is_monotonic',
                'monotonicity_direction',
                'chi_square_statistic',
                'chi_square_p_value',
                'bin_balance_gini',
                'min_bin_proportion',
                'entropy',
            ],
            'value': [
                mono_score,
                is_mono,
                direction,
                chi2,
                p_value,
                gini,
                min_prop,
                entropy,
            ]
        })
        
        return results
