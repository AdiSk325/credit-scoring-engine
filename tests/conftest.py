"""Test configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    X = np.random.normal(50, 10, 1000)
    y = (X > 50).astype(int)
    return X, y


@pytest.fixture
def sample_categorical_data():
    """Generate sample categorical data."""
    np.random.seed(42)
    X = np.random.choice(['A', 'B', 'C', 'D'], 1000)
    y = np.random.randint(0, 2, 1000)
    return X, y
