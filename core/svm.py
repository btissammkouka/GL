import numpy as np
from sklearn.svm import SVC
from .base_model import BaseModel


class SVMModel(BaseModel):
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=42, **kwargs):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True,  # Enable probability estimates for predict_proba
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
    
    def get_support_vectors(self):
        self._validate_fitted()
        return self.model.support_vectors_
    
    def get_n_support(self):
        self._validate_fitted()
        return self.model.n_support_

