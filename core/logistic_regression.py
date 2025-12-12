import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    
    def __init__(self, max_iter=1000, random_state=42, **kwargs):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = LogisticRegression(
            max_iter=self.max_iter,
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
    
    def get_coefficients(self):
        self._validate_fitted()
        return self.model.coef_[0]
    
    def get_intercept(self):
        self._validate_fitted()
        return self.model.intercept_[0]

