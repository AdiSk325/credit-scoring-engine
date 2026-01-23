"""Unit tests for binning module."""

import pytest
import numpy as np
import pandas as pd
from credit_scoring.binning import ContinuousBinning, CategoricalBinning, BinningEvaluator


class TestContinuousBinning:
    """Test cases for ContinuousBinning."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.normal(50, 10, 1000)
        self.y = (self.X > 50).astype(int)
    
    def test_quantile_binning(self):
        """Test quantile binning method."""
        binner = ContinuousBinning(method='quantile', n_bins=5)
        X_binned = binner.fit_transform(self.X, self.y)
        
        assert len(X_binned) == len(self.X)
        assert binner.bins_ is not None
        assert len(binner.bins_) == 4  # n_bins - 1 boundaries
        assert len(binner.bin_labels_) == 5
    
    def test_equal_width_binning(self):
        """Test equal width binning method."""
        binner = ContinuousBinning(method='equal_width', n_bins=5)
        X_binned = binner.fit_transform(self.X, self.y)
        
        assert len(X_binned) == len(self.X)
        assert binner.bins_ is not None
    
    def test_tree_binning(self):
        """Test tree-based binning method."""
        binner = ContinuousBinning(method='tree', max_depth=3)
        X_binned = binner.fit_transform(self.X, self.y)
        
        assert len(X_binned) == len(self.X)
        assert binner.bins_ is not None
    
    def test_custom_binning(self):
        """Test custom binning method."""
        custom_boundaries = [30, 40, 50, 60, 70]
        binner = ContinuousBinning(method='custom', custom_boundaries=custom_boundaries)
        X_binned = binner.fit_transform(self.X)
        
        assert len(X_binned) == len(self.X)
        assert np.array_equal(binner.bins_, custom_boundaries)
    
    def test_bin_statistics(self):
        """Test bin statistics calculation."""
        binner = ContinuousBinning(method='quantile', n_bins=5)
        binner.fit(self.X, self.y)
        
        stats = binner.get_bin_statistics(self.X, self.y)
        
        assert isinstance(stats, pd.DataFrame)
        assert 'bin' in stats.columns
        assert 'count' in stats.columns
        assert 'event_rate' in stats.columns
        assert len(stats) == 5


class TestCategoricalBinning:
    """Test cases for CategoricalBinning."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        self.y = np.random.randint(0, 2, 1000)
    
    def test_keep_all(self):
        """Test keep all categories method."""
        binner = CategoricalBinning(method='keep_all')
        X_binned = binner.fit_transform(self.X, self.y)
        
        assert len(X_binned) == len(self.X)
        assert len(binner.categories_) == 5
    
    def test_rare_grouping(self):
        """Test rare category grouping."""
        binner = CategoricalBinning(method='rare', rare_threshold=0.1)
        X_binned = binner.fit_transform(self.X, self.y)
        
        assert len(X_binned) == len(self.X)
        # Category E (5%) should be grouped as RARE
        assert 'RARE' in binner.categories_
    
    def test_custom_mapping(self):
        """Test custom category mapping."""
        custom_mapping = {'A': 'GROUP1', 'B': 'GROUP1', 'C': 'GROUP2', 'D': 'GROUP2', 'E': 'GROUP3'}
        binner = CategoricalBinning(method='custom', custom_mapping=custom_mapping)
        X_binned = binner.fit_transform(self.X)
        
        assert len(X_binned) == len(self.X)
        assert set(X_binned) == {'GROUP1', 'GROUP2', 'GROUP3'}


class TestBinningEvaluator:
    """Test cases for BinningEvaluator."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_binned = np.random.randint(0, 5, 1000)
        self.y = np.random.randint(0, 2, 1000)
    
    def test_monotonicity(self):
        """Test monotonicity evaluation."""
        evaluator = BinningEvaluator()
        score, is_mono, direction = evaluator.evaluate_monotonicity(self.X_binned, self.y)
        
        assert 0 <= score <= 1
        assert isinstance(is_mono, bool)
        assert direction in ['increasing', 'decreasing', 'non-monotonic']
    
    def test_chi_square(self):
        """Test chi-square test."""
        evaluator = BinningEvaluator()
        chi2, p_value = evaluator.chi_square_test(self.X_binned, self.y)
        
        assert chi2 >= 0
        assert 0 <= p_value <= 1
    
    def test_bin_balance(self):
        """Test bin balance calculation."""
        evaluator = BinningEvaluator()
        gini, min_prop = evaluator.calculate_bin_balance(self.X_binned)
        
        assert 0 <= gini <= 1
        assert 0 <= min_prop <= 1
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive binning quality evaluation."""
        evaluator = BinningEvaluator()
        results = evaluator.evaluate_binning_quality(self.X_binned, self.y)
        
        assert isinstance(results, pd.DataFrame)
        assert 'metric' in results.columns
        assert 'value' in results.columns
        assert len(results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
