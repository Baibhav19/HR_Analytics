import os
import yaml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

class load_and_split_data:
    
    def get_data_func(self, data_path):
        df = pd.read_csv(data_path, sep=',')
        return df

    def split_train_test_func(self, config_path):
        train_path = config_path['split_data']["train_path"]
        test_path = config_path['split_data']["test_path"]
        random_state = config_path['base']["random_state"]
        load_path = config_path['load_data']["raw_dataset_csv"]
        df = pd.read_csv(load_path, sep=",")
        test_size = config_path['split_data']["test_size"]
        train, test = train_test_split(df, random_state = random_state, test_size=test_size)
        train.to_csv(train_path, index=False, sep=',')
        test.to_csv(test_path, index=False, sep=',')

    def split_raw_data(self, df, config_path):
        
        df.to_csv(config_path, index=False, sep=',')

    def __init__(self, path=''):
        if(path == ''):
            self.path = 'params.yaml'
        self.raw_data = dict()
        with open(self.path) as f:
            self.raw_data  = yaml.safe_load(f)
        print(self.raw_data)
        df = self.get_data_func(self.raw_data['data_source']['s3_source'])
        self.split_raw_data(df, self.raw_data['load_data']['raw_dataset_csv'])
        self.split_train_test_func(self.raw_data)

if __name__ == "__main__":
	load_and_split_data()
		



