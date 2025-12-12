import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Union, Optional, Dict, Any
from core.base_model import BaseModel
from core.dataset import RespiratoryInfectionDataset


class Validator:
    
    def __init__(self, models_dir='Models'):
        self.models_dir = models_dir
        self.model = None
        self.is_loaded = False
    
    def load_model(self, model_path: str = None, model: BaseModel = None):
        if model_path and model:
            raise ValueError("Provide either model_path or model, not both.")
        
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            loaded = joblib.load(model_path)
            if not isinstance(loaded, BaseModel):
                raise TypeError("Loaded model must be an instance of BaseModel")
            
            self.model = loaded
            self.is_loaded = True
            print(f"Model loaded from: {model_path}")
            print(f"Model type: {self.model.__class__.__name__}")
        
        elif model:
            if not isinstance(model, BaseModel):
                raise TypeError("Model must be an instance of BaseModel")
            
            if not model.is_fitted:
                raise ValueError("Model must be trained before inference. Call fit() first.")
            
            self.model = model
            self.is_loaded = True
            print(f"Model loaded: {self.model.__class__.__name__}")
        
        else:
            raise ValueError("Must provide either model_path or model")
        
        return self
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        self._validate_loaded()
        X = self._validate_input(X)
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        self._validate_loaded()
        X = self._validate_input(X)
        
        try:
            probabilities = self.model.predict_proba(X)
            return probabilities
        except NotImplementedError:
            raise NotImplementedError("This model does not support predict_proba")
    
    def validate(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], 
                 verbose: bool = True) -> Dict[str, Any]:
        self._validate_loaded()
        X = self._validate_input(X)
        y = np.asarray(y)
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        }
        
        if verbose:
            self._print_metrics(metrics, y, predictions)
        
        return metrics
    
    def validate_with_dataset(self, csv_path: str = None, 
                             dataset: RespiratoryInfectionDataset = None,
                             verbose: bool = True) -> Dict[str, Any]:
        if csv_path and dataset:
            raise ValueError("Provide either csv_path or dataset, not both.")
        
        if csv_path:
            dataset = RespiratoryInfectionDataset(csv_path)
            dataset.load()
            dataset.prepare()
        elif not dataset:
            raise ValueError("Must provide either csv_path or dataset")
        
        X, y = dataset.get_data()
        return self.validate(X, y, verbose=verbose)
    
    def predict_single(self, patient_data: Union[np.ndarray, list], 
                      return_proba: bool = False) -> Union[int, Dict[str, Any]]:
        self._validate_loaded()
        
        X = np.asarray(patient_data)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        prediction = self.predict(X)[0]
        
        if return_proba:
            try:
                proba = self.predict_proba(X)[0]
                return {
                    'prediction': int(prediction),
                    'probability': {
                        'class_0': float(proba[0]),
                        'class_1': float(proba[1])
                    },
                    'confidence': float(proba[1])
                }
            except NotImplementedError:
                return {
                    'prediction': int(prediction),
                    'probability': None,
                    'confidence': None
                }
        
        return int(prediction)
    
    def _validate_loaded(self):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
    
    def _validate_input(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if X is None:
            raise ValueError("Input X cannot be None")
        
        X = np.asarray(X)
        if len(X) == 0:
            raise ValueError("Input X cannot be empty")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return X
    
    def _print_metrics(self, metrics: Dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray):
        print(f"\n{'='*60}")
        print("Model Validation Results")
        print(f"{'='*60}")
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                 Predicted")
        print(f"                 No    Yes")
        print(f"Actual No    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Yes   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Not Infected', 'Infected']))
        print(f"{'='*60}")
    
        return self.model


