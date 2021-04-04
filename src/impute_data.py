from sklearn.impute import SimpleImputer

class Impute_Missing_Data():
    
    def __init__(self, cat_var, num_var):
        self.cat_var = cat_var
        self.num_var = num_var
        self.sin = SimpleImputer(strategy='mean')
        self.sic = SimpleImputer(strategy='most_frequent')
    
    def fit(self, x_dataset, y=None):
        self.sic.fit(x_dataset[self.cat_var])
        self.sin.fit(x_dataset[self.num_var])
        return self

    def transform(self, x_dataset, y=None):
        x_dataset.loc[:][self.cat_var] = self.sic.transform(x_dataset[self.cat_var])
        x_dataset.loc[:][self.num_var] = self.sin.transform(x_dataset[self.num_var])
        x_dataset['KPI_and_training_Score'].fillna(value=0, inplace=True)
        #x_dataset['awards_won_gby_reg_dep'].fillna(value=0, inplace=True)
        return x_dataset