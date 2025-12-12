import sys
import os
import pytest
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.trainer import Trainer
from core.dataset import RespiratoryInfectionDataset

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from core.decision_tree import DecisionTreeModel

def test_imports():
    """Test that main modules can be imported."""
    assert Trainer is not None
    assert RespiratoryInfectionDataset is not None

def test_dataset_preparation():
    """Test RespiratoryInfectionDataset preparation."""
    # Create dummy data
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'infected': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Mock pd.read_csv to return dummy data
    with patch('pandas.read_csv', return_value=df):
        dataset = RespiratoryInfectionDataset(csv_path='dummy.csv')
        dataset.load()
        dataset.prepare()
        
        X, y = dataset.get_data()
        feature_names = dataset.get_feature_names()
        
        assert X.shape == (5, 2)
        assert y.shape == (5,)
        assert len(feature_names) == 2
        assert 'infected' not in feature_names
        assert 'feature1' in feature_names

def test_decision_tree_model():
    """Test DecisionTreeModel training and prediction."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    model = DecisionTreeModel(max_depth=2)
    model.fit(X, y)
    
    assert model.is_fitted
    
    predictions = model.predict(X)
    assert len(predictions) == 4
    
    # Check that it learned something (accuracy should be decent on training data)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.5

def test_trainer_flow(tmp_path):
    """Test Trainer training and saving flow."""
    models_dir = tmp_path / "Models"
    trainer = Trainer(models_dir=str(models_dir))
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeModel()
    
    trained_model, model_path = trainer.train(model, X, y, model_name="test_model")
    
    assert trained_model.is_fitted
    assert os.path.exists(model_path)
    assert "test_model" in model_path
    
    # Test loading
    loaded_model = trainer.load_model(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, DecisionTreeModel)

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_imports()
        print("test_imports passed")
        
        test_dataset_preparation()
        print("test_dataset_preparation passed")
        
        test_decision_tree_model()
        print("test_decision_tree_model passed")
        
        # For test_trainer_flow, we need a tmp_path. 
        # In a real script run we might skip it or use a temp dir.
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp_dir:
             from pathlib import Path
             test_trainer_flow(Path(tmp_dir))
        print("test_trainer_flow passed")
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
