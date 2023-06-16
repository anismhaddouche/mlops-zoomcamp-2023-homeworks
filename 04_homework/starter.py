#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd

import sys 




def read_data(filename):
    print(f'Reading the data file {filename}...')
    
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df




def get_predictions(year, month): 
    print(f'opening the model ...')

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet')
    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    print(f'Get predictions...')

    y_pred = pd.Series(model.predict(X_val))
    df_result = df.copy()
    df_result["predictions"]  = y_pred
    path = f'prediction_{year}_{month}.parquet'
    df_result.to_parquet(path)
    print(y_pred.describe())
    return path

def upload_gcp(path:str):
    
    return None

def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    path = get_predictions(year=year,month=month)
    upload_gcp(path)

if __name__ == '__main__':
    run()


















