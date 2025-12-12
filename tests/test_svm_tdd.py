import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_svm_functionality():
    """Test SVM Model functionality (TDD)."""
    # This import should fail if the model is not implemented yet
    from core.svm import SVMModel
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    model = SVMModel(kernel='linear', C=1.0)
    model.fit(X, y)
    
    assert model.is_fitted
    
    predictions = model.predict(X)
    assert len(predictions) == 4
    
    # Check accuracy
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.5
