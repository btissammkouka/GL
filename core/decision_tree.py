import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, **kwargs):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
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
    
    def get_max_depth(self):
        return self.max_depth
    
    def get_tree_depth(self):
        self._validate_fitted()
        return self.model.tree_.max_depth

