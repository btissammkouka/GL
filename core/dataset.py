import pandas as pd
import numpy as np


class RespiratoryInfectionDataset:
    
    def __init__(self, csv_path='data/clclinical_respiratory_infection_dataset.csv'):
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_name = 'infected'
        
    def load(self):
        self.data = pd.read_csv(self.csv_path)
        print(f"Dataset loaded successfully from: {self.csv_path}")
        print(f"Dataset shape: {self.data.shape}")
        
    def prepare(self):
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        self.y = self.data[self.target_name].values
        self.X = self.data.drop(columns=[self.target_name]).values
        self.feature_names = self.data.drop(columns=[self.target_name]).columns.tolist()
        
        print(f"\nFeatures shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"\nNumber of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        print(f"\nTarget distribution:")
        unique, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({count/len(self.y)*100:.1f}%)")
        
    def get_data(self):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not prepared. Call prepare() first.")
        
        return self.X, self.y
    
    def get_feature_names(self):
        if self.feature_names is None:
            raise ValueError("Dataset not prepared. Call prepare() first.")
        
        return self.feature_names

