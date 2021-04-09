import os
import yaml
import pandas as pd
import numpy as np
import joblib
import json

class prediction:
    
    def __init__(self, type="normal"):
        self.raw_data = dict()
        self.path = 'params.yaml'
        with open(self.path) as f:
            self.raw_data  = yaml.safe_load(f)
        random_state = self.raw_data["base"]["random_state"]
        model_dir = self.raw_data["model_dir"]
        model_path = os.path.join(model_dir, "model.joblib")
        pipeline_path = os.path.join(model_dir, "pipeline_hr.joblib")
        test_data_path = self.raw_data["test_data"]["test_data_csv"]
        test_result_path = self.raw_data["test_data"]["test_results"]
        target = self.raw_data["base"]["target_col"]
        classification_threshold = self.raw_data["classification_threshold"]
        self.result = self.predict_func(test_data_path, test_result_path, model_path, pipeline_path, target, classification_threshold)

    def predict_func(self, test_data_path, test_result_path, model_path, pipeline_path, target, classification_threshold):
        model = joblib.load(model_path)
        pip_hr = joblib.load(pipeline_path)
        X_test = pd.read_csv(test_data_path)
        X_test_transformed = pip_hr.transform(X_test)
        test_pred_prob = model.predict_proba(X_test_transformed)
        
        y_test_pred_th = (test_pred_prob[:,0] <= (classification_threshold/100)).astype('int')
        X_test[target] = y_test_pred_th
        pd.DataFrame(X_test).to_csv(test_result_path)
        return "200 OK"

if __name__ == "__main__":
    pred = prediction()