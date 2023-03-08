import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml
import sys
from src.utils import create_folder
import xgboost as xgb
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, plotting, space_eval
from src.utils import *
from src.metrics import *
from datetime import timedelta, datetime
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import json
import pickle
import matplotlib.pyplot as plt


#Input State Name
parser = argparse.ArgumentParser(description='xgboost simulation model')
parser.add_argument('--state_name', type=str, default='TX', help='the name of state in USA')
args = parser.parse_args()
print(args.state_name)


proj_path = Path.cwd()
with open(os.path.join(proj_path, 'catalog.yml'), "r") as f:
    catalog = yaml.safe_load(f)['breakfast']
    
with open(os.path.join(proj_path, 'params.yml'), "r") as f:
    params = yaml.safe_load(f)



merged_data = pd.read_csv('data/processed/merged_data.csv')
merged_data['WEEK_END_DATE'] = pd.to_datetime(merged_data['WEEK_END_DATE'])
original_data = merged_data.copy()
merged_data['WEEK_END_DATE'] = merged_data['WEEK_END_DATE'] + timedelta(days=3)
date_ranges = make_dates(params['breakfast']['experiment_dates'])
print('date_ranges', date_ranges)

stores = list(params['breakfast']['dataset']['store_ids'].keys())
upcs = list(params['breakfast']['dataset']['upc_ids'].keys())
import itertools
store_upc_pairs = list(itertools.product(stores, upcs))
print(store_upc_pairs)

#for store_id, upc_id in store_upc_pairs:
#    create_folder(os.path.join(proj_path, 'runs'))
for _, train_start, train_end, valid_start, valid_end, test_start, test_end in date_ranges.itertuples():
    lag_units = params['xgb']['window_size']
    avg_units = params['xgb']['avg_units']
    #control features

    #filtered_data = merged_data[merged_data['ADDRESS_STATE_PROV_CODE']=='TX'][['WEEK_END_DATE', 'STORE_NUM', 'UPC', 'UNITS', 'PRICE', 'BASE_PRICE', 'DESCRIPTION', 'MANUFACTURER', 'CATEGORY', 'SUB_CATEGORY', 'PRODUCT_SIZE', 'STORE_ID', 'STORE_NAME', 'ADDRESS_CITY_NAME', 'MSA_CODE', 'SEG_VALUE_NAME', 'PARKING_SPACE_QTY', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']].copy()
    #filtered_data = merged_data[merged_data['ADDRESS_STATE_PROV_CODE']==args.state_name][['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'STORE_NUM', 'UPC', 'UNITS', 'PRICE', 'BASE_PRICE', 'DESCRIPTION', 'MANUFACTURER', 'CATEGORY', 'SUB_CATEGORY', 'PRODUCT_SIZE', 'STORE_ID', 'STORE_NAME', 'ADDRESS_CITY_NAME','MSA_CODE', 'SEG_VALUE_NAME', 'PARKING_SPACE_QTY', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']].copy()
    filtered_data = merged_data[merged_data['ADDRESS_STATE_PROV_CODE']==args.state_name][['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'STORE_NUM', 'UPC', 'UNITS', 'PRICE', 'BASE_PRICE', 'STORE_ID', 'MSA_CODE', 'PARKING_SPACE_QTY', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']].copy()

    previous_filtered_data = filtered_data.copy()
    print(list(previous_filtered_data))
    for i in range(1,9):
        for column in list(previous_filtered_data):
            filtered_data[column + '_' + str(i)] = filtered_data[column].shift(i)
        filtered_data.pop('WEEK_END_DATE_' + str(i))
    print(list(filtered_data))

    #Filter data 
    #make_lag_features(filtered_data, lag_units, col_name='UNITS', prefix_name='lag-units', inplace=True)
    #make_historical_avg(filtered_data, r_list=avg_units, col_n='lag-units-1', google_trends=True)
    add_datepart(filtered_data, fldname='WEEK_END_DATE', drop=False)

    train_df = filtered_data[(filtered_data['WEEK_END_DATE']>=train_start) & (filtered_data['WEEK_END_DATE']<=train_end)].copy()
    valid_df = filtered_data[(filtered_data['WEEK_END_DATE']>=valid_start) & (filtered_data['WEEK_END_DATE']<=valid_end)].copy()
    test_df = filtered_data[(filtered_data['WEEK_END_DATE']>=test_start) & (filtered_data['WEEK_END_DATE']<=test_end)].copy()

    train_df.set_index('WEEK_END_DATE', inplace=True)
    valid_df.set_index('WEEK_END_DATE', inplace=True)
    test_df.set_index('WEEK_END_DATE', inplace=True)

    X_train = train_df
    y_train = X_train.pop('UNITS')
    X_valid = valid_df
    y_valid = X_valid.pop('UNITS')
    X_test = test_df
    y_test = X_test.pop('UNITS')

    xgb_model = pickle.load(open('./xgboost_models/' + str(test_end.date()) + '{}_xgboost.pickle.dat'.format(args.state_name), "rb"))
    
    raw_fname = './new_results/' + str(test_end.date()) + 'regress_12weeks_xgb_{}.csv'.format(args.state_name)
    raw_data = pd.read_csv(raw_fname)

    X_test['PRICE'] = X_test['PRICE'] - 1
    fig = plt.figure()
    plt.title(str(test_end.date()) + '{}_intervene_experiment'.format(args.state_name))
    plt.plot(raw_data['y_true'].values, '.-', label='y_true')

    for i in range(0, 7):
        X_test['PRICE'] = X_test['PRICE'] + 0.5
        test_preds = xgb_model.predict(X_test)
        plt.plot(test_preds, '.-', label='delta {}'.format( -1 + i * 0.5))

    plt.legend()
    plt.grid()
    plt.savefig('./figs_label/' + str(test_end.date()) + '{}_intervene_experiment.png'.format(args.state_name))
print(args.state_name + ' XGBoost finished')