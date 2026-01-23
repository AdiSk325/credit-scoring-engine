"""Unit tests for metrics module."""

import pytest
import numpy as np
from credit_scoring.metrics import (
    gini_coefficient,
    ks_statistic,
    population_stability_index,
    hosmer_lemeshow_test,
)


class TestPerformanceMetrics:
    """Test cases for performance metrics."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 1000)
        self.y_pred = np.random.random(1000)
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        gini = gini_coefficient(self.y_true, self.y_pred)
        
        assert isinstance(gini, float)
        assert -1 <= gini <= 1
    
    def test_ks_statistic(self):
        """Test KS statistic calculation."""
        ks, threshold = ks_statistic(self.y_true, self.y_pred)
        
        assert isinstance(ks, float)
        assert 0 <= ks <= 1
        assert isinstance(threshold, (int, float, np.number))
    
    def test_perfect_separation(self):
        """Test metrics with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        gini = gini_coefficient(y_true, y_pred)
        assert gini == 1.0


class TestStabilityMetrics:
    """Test cases for stability metrics."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.expected = np.random.normal(50, 10, 1000)
        self.actual_stable = np.random.normal(50, 10, 1000)
        self.actual_shifted = np.random.normal(55, 10, 1000)
    
    def test_psi_stable(self):
        """Test PSI with stable distribution."""
        psi = population_stability_index(self.expected, self.actual_stable)
        
        assert isinstance(psi, float)
        assert psi >= 0
        assert psi < 0.1  # Should be low for similar distributions
    
    def test_psi_shifted(self):
        """Test PSI with shifted distribution."""
        psi = population_stability_index(self.expected, self.actual_shifted)
        
        assert isinstance(psi, float)
        assert psi > 0
        # Should be higher for shifted distributions
    
    def test_psi_bins(self):
        """Test PSI with different number of bins."""
        psi_5 = population_stability_index(self.expected, self.actual_stable, bins=5)
        psi_20 = population_stability_index(self.expected, self.actual_stable, bins=20)
        
        assert isinstance(psi_5, float)
        assert isinstance(psi_20, float)


class TestCalibrationMetrics:
    """Test cases for calibration metrics."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 1000)
        self.y_pred = np.random.random(1000)
    
    def test_hosmer_lemeshow(self):
        """Test Hosmer-Lemeshow test."""
        chi2, p_value = hosmer_lemeshow_test(self.y_true, self.y_pred)
        
        assert isinstance(chi2, float)
        assert chi2 >= 0
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_hosmer_lemeshow_bins(self):
        """Test Hosmer-Lemeshow with different bin counts."""
        chi2_5, p_value_5 = hosmer_lemeshow_test(self.y_true, self.y_pred, n_bins=5)
        chi2_10, p_value_10 = hosmer_lemeshow_test(self.y_true, self.y_pred, n_bins=10)
        
        assert isinstance(chi2_5, float)
        assert isinstance(chi2_10, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
