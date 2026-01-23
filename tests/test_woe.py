"""Unit tests for WoE module."""

import pytest
import numpy as np
import pandas as pd
from credit_scoring.woe import WoEEncoder, InformationValue, DScore


class TestWoEEncoder:
    """Test cases for WoEEncoder."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_binned = np.random.randint(0, 5, 1000)
        self.y = np.random.randint(0, 2, 1000)
    
    def test_fit_transform(self):
        """Test WoE fit and transform."""
        encoder = WoEEncoder(smooth=0.5)
        X_woe = encoder.fit_transform(self.X_binned, self.y)
        
        assert len(X_woe) == len(self.X_binned)
        assert encoder.woe_mapping_ is not None
        assert len(encoder.woe_mapping_) == 5
    
    def test_woe_stats(self):
        """Test WoE statistics."""
        encoder = WoEEncoder()
        encoder.fit(self.X_binned, self.y)
        
        stats = encoder.get_woe_stats()
        
        assert isinstance(stats, pd.DataFrame)
        assert 'bin' in stats.columns
        assert 'woe' in stats.columns
        assert 'event_rate' in stats.columns
    
    def test_missing_values(self):
        """Test WoE with missing values."""
        X_with_nan = self.X_binned.astype(float)
        X_with_nan[::10] = np.nan
        
        encoder = WoEEncoder(handle_missing=True)
        X_woe = encoder.fit_transform(X_with_nan, self.y)
        
        assert len(X_woe) == len(X_with_nan)
        assert '__MISSING__' in encoder.woe_mapping_
    
    def test_binary_target_validation(self):
        """Test that encoder validates binary target."""
        encoder = WoEEncoder()
        y_invalid = np.random.randint(0, 3, 1000)  # Non-binary
        
        with pytest.raises(ValueError):
            encoder.fit(self.X_binned, y_invalid)


class TestInformationValue:
    """Test cases for InformationValue."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_binned = np.random.randint(0, 5, 1000)
        self.y = np.random.randint(0, 2, 1000)
    
    def test_calculate_iv(self):
        """Test IV calculation."""
        iv_calc = InformationValue()
        iv_score = iv_calc.calculate(self.X_binned, self.y)
        
        assert isinstance(iv_score, float)
        assert iv_score >= 0
    
    def test_iv_stats(self):
        """Test IV statistics."""
        iv_calc = InformationValue()
        iv_calc.calculate(self.X_binned, self.y)
        
        stats = iv_calc.get_iv_stats()
        
        assert isinstance(stats, pd.DataFrame)
        assert 'bin' in stats.columns
        assert 'iv' in stats.columns
        assert 'woe' in stats.columns
    
    def test_interpret_iv(self):
        """Test IV interpretation."""
        iv_calc = InformationValue()
        
        assert "Not useful" in iv_calc.interpret_iv(0.01)
        assert "Weak" in iv_calc.interpret_iv(0.05)
        assert "Medium" in iv_calc.interpret_iv(0.15)
        assert "Strong" in iv_calc.interpret_iv(0.35)
        assert "Suspicious" in iv_calc.interpret_iv(0.6)
    
    def test_calculate_multiple(self):
        """Test IV calculation for multiple features."""
        X_df = pd.DataFrame({
            'feature1': np.random.randint(0, 5, 1000),
            'feature2': np.random.randint(0, 3, 1000),
        })
        
        iv_calc = InformationValue()
        iv_summary = iv_calc.calculate_multiple(X_df, self.y)
        
        assert isinstance(iv_summary, pd.DataFrame)
        assert 'feature' in iv_summary.columns
        assert 'iv' in iv_summary.columns
        assert len(iv_summary) == 2


class TestDScore:
    """Test cases for DScore."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create data with clear separation
        self.X_events = np.random.normal(10, 2, 500)
        self.X_non_events = np.random.normal(8, 2, 500)
        self.X = np.concatenate([self.X_events, self.X_non_events])
        self.y = np.concatenate([np.ones(500), np.zeros(500)])
    
    def test_calculate_dscore(self):
        """Test d-score calculation."""
        dscore = DScore.calculate(self.X, self.y)
        
        assert isinstance(dscore, float)
        assert dscore > 0  # Should be positive given our data
    
    def test_interpret_dscore(self):
        """Test d-score interpretation."""
        interpretation = DScore.interpret_dscore(0.1)
        assert "Small" in interpretation
        
        interpretation = DScore.interpret_dscore(0.3)
        assert "Medium" in interpretation
        
        interpretation = DScore.interpret_dscore(0.6)
        assert "Large" in interpretation
        
        interpretation = DScore.interpret_dscore(1.0)
        assert "Very large" in interpretation
    
    def test_calculate_with_stats(self):
        """Test d-score calculation with statistics."""
        stats = DScore.calculate_with_stats(self.X, self.y)
        
        assert isinstance(stats, dict)
        assert 'dscore' in stats
        assert 'mean_events' in stats
        assert 'mean_non_events' in stats
        assert 'pooled_std' in stats
    
    def test_calculate_multiple(self):
        """Test d-score for multiple features."""
        X_df = pd.DataFrame({
            'feature1': self.X,
            'feature2': np.random.normal(0, 1, 1000),
        })
        
        dscore_summary = DScore.calculate_multiple(X_df, self.y)
        
        assert isinstance(dscore_summary, pd.DataFrame)
        assert 'feature' in dscore_summary.columns
        assert 'dscore' in dscore_summary.columns
        assert len(dscore_summary) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
