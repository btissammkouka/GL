import os
import joblib
import numpy as np
from datetime import datetime
from core.base_model import BaseModel


class Trainer:
    
    def __init__(self, models_dir='Models'):
        self.models_dir = models_dir
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Created directory: {self.models_dir}")
    
    def train(self, model, X, y, model_name=None, save_model=True):
        if X is None or y is None:
            raise ValueError("X and y cannot be None. Provide training data.")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty.")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
        
        if not isinstance(model, BaseModel):
            raise TypeError("Model must be an instance of BaseModel")
        
        print(f"\n{'='*60}")
        print(f"Training model: {model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Training samples: {len(X)}")
        
        model.fit(X, y)
        
        print(f"Model trained successfully!")
        
        model_path = None
        if save_model:
            if model_name is None:
                model_name = model.__class__.__name__
            model_path = self.save_model(model, model_name)
        
        return model, model_path
    
    def save_model(self, model, model_name):
        if not model.is_fitted:
            raise ValueError("Model must be trained before saving. Call fit() first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(model, filepath)
        
        print(f"\nModel saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return model



