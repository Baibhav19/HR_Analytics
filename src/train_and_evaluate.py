# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, RandomOverSampler, SMOTE 
from scale import Scale
from impute_data import Impute_Missing_Data
from generate_new_feature import CalculateNewFeature
from transform_skewed_cols import transform_skewed_data
import joblib
import json
import yaml
'''Class to train and evaluate.
Also write metrics to JSON file.
'''
class train_and_evaluate:

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1_scr = f1_score(actual, pred)
        return accuracy, f1_scr

    def train_and_evaluate_func(self, config):
        #config = read_params(config_path)
        test_data_path = config["split_data"]["test_path"]
        train_data_path = config["split_data"]["train_path"]
        random_state = config["base"]["random_state"]
        model_dir = config["model_dir"]
        classification_threshold = config["classification_threshold"]
        n_estimators = config["estimators"]["RandomForestClassifier"]["n_estimators"]
        max_depth = config["estimators"]["RandomForestClassifier"]["max_depth"]
        target = [config["base"]["target_col"]]
        print("target")
        train = pd.read_csv(train_data_path, sep=",")
        test = pd.read_csv(test_data_path, sep=",")
        
        y_train = train[target]
        y_test = test[target]

        cat_features = []
        num_features = []
        for col in train.columns:
            if (train[col].dtype =='O'):
                cat_features.append(col)
            else:
                if (train[col].unique().size > 10):
                    num_features.append(col)
                else:
                    cat_features.append(col)
        
            skew_data = train[num_features].skew(axis = 0, skipna = True)
            skewed_cols = skew_data[skew_data > 1.1].index.values

        X_train = train.drop(target, axis=1)    
        X_test = test.drop(target, axis=1)
        
        pre_process = ColumnTransformer(remainder='passthrough', 
                                transformers=[
                                              ('one_hot_encoding', ce.OneHotEncoder(), ['gender','education','recruitment_channel']),
                                              #('binary_encoding', ce.BinaryEncoder(), ['department', 'region']),
                                              ('drop_col', 'drop', [ 'gender', 
                                                                    'employee_id', 'recruitment_channel',
                                                                    'avg_training_score', 'education','department', 
                                                                    'region', 'KPIs_score_gby_reg_dep', 'KPIs_met >80%', 'length_of_service'])
                                             ])
        num_var = ['previous_year_rating']
        cat_var = ['education']

        print('Initializing Pipeline')

        pip_sm =  Pipeline(steps=[('calculate_new_feature', CalculateNewFeature()),
                     ('impute_missing', Impute_Missing_Data(cat_var=cat_var, num_var=num_var)),
                     ('trans_skew', transform_skewed_data(skew_cols=skewed_cols)),
                     ('pre_process', pre_process),
                     ('scale', Scale())
                     ])
        
        X_train_smote = pip_sm.fit_transform(X_train)
        #print(X_train_smote)
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_smote, y_smote = smote.fit_resample(X_train_smote, y_train)
        X_test_smote = pip_sm.transform(X_test)
        #print(X_test_smote)
        rf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
        
        rf.fit(X_smote, y_smote.values)
        
        pred_prob = rf.predict_proba(X_test_smote)
        predicted_qualities = (pred_prob[:,0] <= (classification_threshold/100)).astype('int')
        
        (accuracy, f1_score) = self.eval_metrics(y_test.values, predicted_qualities)

        
        print("  Accuracy: %s" % accuracy)
        print("  f1 score: %s" % f1_score)
        print(confusion_matrix(y_test.values, predicted_qualities))
    #####################################################
        scores_file = config["reports"]["scores"]
        #params_file = config["reports"]["params"]

        with open(scores_file, "w") as f:
            scores = {
                "accuracy": accuracy,
                "f1_score": f1_score
            }
            json.dump(scores, f, indent=4)

        
    #####################################################


        os.makedirs(model_dir, exist_ok=True)
        pipeline_path = os.path.join(model_dir, "pipeline_hr.joblib")
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(pip_sm, pipeline_path)
        joblib.dump(rf, model_path)
        
    def __init__(self):
        self.raw_data = dict()
        self.path = 'params.yaml'
        with open(self.path) as f:
            self.raw_data  = yaml.safe_load(f)
        #print("as")
        self.train_and_evaluate_func(self.raw_data)
        

if __name__ == "__main__":
    train_and_evaluate()