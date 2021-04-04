from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Scale():
    
    def __init__(self):
        self.minmax_scale = MinMaxScaler()
    
    def fit(self, x_dataset, y=None):
        self.minmax_scale.fit(x_dataset)
        return self

    def transform(self, x_dataset, y=None):
        return self.minmax_scale.transform(x_dataset)