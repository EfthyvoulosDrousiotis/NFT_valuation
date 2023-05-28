# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:20:05 2023

@author: efthi
"""

#import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

label_encoder = LabelEncoder()


# Assuming you have your features (X) and target variable (y) ready
Etherium_price = pd.read_csv(r"C:\Users\efthi\Downloads\cryptopunks_test_bundle\20230509\eth_usd_fx_rates.csv")
tokens_metadate = pd.read_csv(r"C:\Users\efthi\Downloads\cryptopunks_test_bundle\20230509\token_metadata.csv")
tokens_sales = pd.read_csv(r"C:\Users\efthi\Downloads\cryptopunks_test_bundle\20230509\token_sales.csv")


'''
Merged and sorted given the token 
index firstly and then the timestamp
'''
Data = pd.merge(tokens_metadate, tokens_sales, on='token_index')
Data = Data.filter(regex='^(?!Unnamed)')


cat_features = ['Skin Tone', "Type", "Hair",
                "Eyewear", "Mouth", "Headwear",
                "Facial Hair", "Smoking Device",
                "Other:Earring", "Neckwear",
                "Skin Feature", "Other:Medical Mask",
                "Other:Clown Nose", "Trait Count",
                "rarest_property_name"]

for feature in cat_features:
    Data[feature] = Data[feature].astype('category').cat.codes


import re
Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


"""
create a dataset with the unique nft which 
are sold only once so we dont have historic data to train our model
"""
# Identify duplicate rows
duplicates = Data.duplicated(subset= "token_index", keep=False)
df_duplicates = Data[duplicates]
df_unique = Data[~duplicates]


'''
This "test" dataset contains only the token 
with the last sold price given the timestamp plus the unique tokens
which might be sold only once
'''
# Sort the dataframe by 'timestamp' in descending order
df_sorted = Data.sort_values(by='timestamp', ascending=False)

# Drop duplicates based on 'token_index' while keeping the row with the largest 'timestamp'
Test_data = df_sorted.drop_duplicates(subset='token_index', keep='first')

X_test = Test_data.drop(columns=["eth",'usd'])
Y_test = Test_data.loc[:, Test_data.columns == 'eth']



'''
Keep the data which have been sold more than once.
'''
Train_data_x = df_sorted[~df_sorted['timestamp'].isin(Test_data['timestamp'])]
Train_data = Train_data_x[~Train_data_x['token_index'].isin(df_unique['token_index'])]

X_train = Train_data.drop(columns=["eth",'usd'])
y_train = Train_data.loc[:, Train_data.columns == 'eth']



# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 100,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the model
num_rounds = 100
model = lgb.train(params, train_data, num_rounds)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, y_pred)
print('Mean Squared Error:', mse)





