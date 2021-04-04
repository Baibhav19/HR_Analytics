import numpy as np

class transform_skewed_data():
    
    def __init__(self, skew_cols):
        self.skew_cols = skew_cols
    
    def fit(self, x_dataset, y=None):
        
        return self

    def transform(self, x_dataset, y=None):
        for col in self.skew_cols:
            x_dataset['log' + col] = np.log(x_dataset[col])
        return x_dataset