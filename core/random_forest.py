import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.kwargs
        )
    
    def fit(self, X, y):
        X = self._validate_input(X)
        y = np.asarray(y)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        self._validate_fitted()
        X = self._validate_input(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self._validate_fitted()
        X = self._validate_input(X)
        
        return self.model.predict_proba(X)
    
    def get_feature_importances(self):
        self._validate_fitted()
        return self.model.feature_importances_
    
    def get_n_estimators(self):
        return self.n_estimators

