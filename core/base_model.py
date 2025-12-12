from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    
    def __init__(self):
        self.is_fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        raise NotImplementedError("predict_proba not implemented for this model")
    
    def get_model(self):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.model
    
    def _validate_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def _validate_input(self, X):
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty.")
        X = np.asarray(X)
        return X

