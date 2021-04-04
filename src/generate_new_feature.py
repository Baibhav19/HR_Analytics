import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CalculateNewFeature():
    
    def __init__(self):
        self.df_region_depart_mean_score = pd.DataFrame()
        self.df_region_depart_KPI_avg_score = pd.DataFrame()
        self.df_region_depart_awards_won = pd.DataFrame()
    
    def fit(self, x_dataset, y=None):
        #self.df_region_depart_mean_score = x_dataset.groupby(['region', 'department'])['avg_training_score'].agg('mean').unstack()
        #self.df_region_depart_los_mean_score = x_dataset.groupby(['region', 'department'])['length_of_service'].agg('mean').unstack()
        return self

    def calculate_score_wrt_reg_and_depart(self, x):
        return x['avg_training_score']/self.df_region_depart_mean_score.loc[x['region']][x['department']]

    def calculate_df_region_depart_KPI_avg_score(self, x):
        try:
            return x['KPIs_met >80%']/self.df_region_depart_KPI_avg_score.loc[x['region']][x['department']]
        except:
            return 0
    
    def calculate_df_region_depart_awards_won(self, x):
        try:
            return x['awards_won?']/self.df_region_depart_awards_won.loc[x['region']][x['department']]
        except:
            return 0
    
    def transform(self, x_dataset, y=None):
        self.df_region_depart_mean_score = x_dataset.groupby(['region', 'department'])['avg_training_score'].agg('mean').unstack()
        self.df_region_depart_KPI_avg_score = x_dataset.groupby(['region', 'department'])['KPIs_met >80%'].agg('mean').unstack()
        self.df_region_depart_awads_won = x_dataset.groupby(['region', 'department'])['awards_won?'].agg('mean').unstack()
        x_dataset = x_dataset.assign(avg_test_score_gby_reg_dep = x_dataset.apply(self.calculate_score_wrt_reg_and_depart, axis=1))
        x_dataset = x_dataset.assign(KPIs_score_gby_reg_dep = x_dataset.apply(self.calculate_df_region_depart_KPI_avg_score, axis=1))
        x_dataset = x_dataset.assign(awards_won_gby_reg_dep = x_dataset.apply(self.calculate_df_region_depart_awards_won, axis=1))
        a = pd.DataFrame(StandardScaler().fit_transform(x_dataset[['avg_test_score_gby_reg_dep', 'KPIs_score_gby_reg_dep']]) , columns=['avg_test_score_gby_reg_dep', 'KPIs_score_gby_reg_dep'])
        a['KPI_and_training_Score'] = a['avg_test_score_gby_reg_dep']+a['KPIs_score_gby_reg_dep'] 
        x_dataset = x_dataset.join(a['KPI_and_training_Score']) 
        return x_dataset
    