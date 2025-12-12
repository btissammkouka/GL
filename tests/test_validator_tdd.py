import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.validator import Validator
from core.base_model import BaseModel

class MockModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.is_fitted = True
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.array([0, 1, 0, 1])
        
    def predict_proba(self, X):
        # Mock probabilities for ROC AUC
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7]
        ])

def test_validator_includes_roc_auc():
    """Test that validator calculates and returns ROC AUC score."""
    validator = Validator()
    model = MockModel()
    validator.load_model(model=model)
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 1, 0, 1])
    
    metrics = validator.validate(X, y, verbose=False)
    
    assert 'roc_auc' in metrics
    assert isinstance(metrics['roc_auc'], float)
    assert 0 <= metrics['roc_auc'] <= 1.0
