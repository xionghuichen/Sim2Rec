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
#parser = argparse.ArgumentParser(description='xgboost simulation model')
#parser.add_argument('--state_name', type=str, default='TX', help='the name of state in USA')
#args = parser.parse_args()
#print(args.state_name)


proj_path = Path.cwd()
with open(os.path.join(proj_path, 'catalog.yml'), "r") as f:
    catalog = yaml.safe_load(f)['breakfast']
    
with open(os.path.join(proj_path, 'params.yml'), "r") as f:
    params = yaml.safe_load(f)

#Hyperparameter Search
space = {
    'eta': hp.quniform('eta', 0.02, 0.5, 0.01),
    'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
    'subsample': hp.quniform('subsample', 0.2, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 1, 0.1),
    'n_estimators': hp.choice('n_estimators', np.arange(5, 150, dtype=int))
}

def optimize():
    
    best = fmin(_score, space, algo=tpe.suggest, trials=trials, max_evals=20, verbose=0)
    return best

store_data = pd.read_csv('data/processed/store.csv')
store_id_list = store_data['STORE_ID'].unique()
store_state = dict()
for store in store_id_list:
    store_state[int(store)] = store_data[store_data['STORE_ID'] == store]['ADDRESS_STATE_PROV_CODE'].values[0]

print('store_state', store_state)
#json_result = json.dumps(store_state)
#with open('data/processed/store_state.json', 'w') as json_file:
#    json_file.write(json_result)


merged_data = pd.read_csv('data/processed/merged_data_norm_3.csv')
merged_data['WEEK_END_DATE'] = pd.to_datetime(merged_data['WEEK_END_DATE'])
train_start = '2009-01-17'
train_end = '2011-01-08'
valid_start = '2011-01-08'
valid_end = '2011-12-31'

#feature_list = ['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'STORE_NUM', 'UPC', 'UNITS', 'PRICE', 'BASE_PRICE', 'STORE_ID', 'MSA_CODE', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']
feature_list = ['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'UNITS', 'PRICE', 'BASE_PRICE']

for store_id in store_id_list:
    filtered_data = merged_data[merged_data['STORE_ID']==store_id][feature_list].copy()

    previous_filtered_data = filtered_data.copy()
    print(list(previous_filtered_data))
    """
    for i in range(1,9):
    for column in list(previous_filtered_data):
        filtered_data[column + '_' + str(i)] = filtered_data[column].shift(i)
    filtered_data.pop('WEEK_END_DATE_' + str(i))
    """


    #add_datepart(filtered_data, fldname='WEEK_END_DATE', drop=False)

    train_df = filtered_data[(filtered_data['WEEK_END_DATE']>=train_start) & (filtered_data['WEEK_END_DATE']<=train_end)].copy()
    valid_df = filtered_data[(filtered_data['WEEK_END_DATE']>=valid_start) & (filtered_data['WEEK_END_DATE']<=valid_end)].copy()


    train_df.set_index('WEEK_END_DATE', inplace=True)
    valid_df.set_index('WEEK_END_DATE', inplace=True)

    print('list(train_df)', list(train_df))

    X_train = train_df
    y_train = X_train.pop('UNITS')
    X_valid = valid_df
    y_valid = X_valid.pop('UNITS')
    print('list(X_train)', list(X_train))
    

    #Function used to perform an evaluation on the validation and return the score to the trained model
    def _score(params):
        xg_boost_model = xgb.XGBRegressor(objective = 'reg:squarederror',
                                        colsample_bytree = params['colsample_bytree'],
                                        learning_rate = params['eta'],
                                        max_depth = params['max_depth'],
                                        min_child_weight = params['min_child_weight'],
                                        n_estimators = params['n_estimators'],
                                        random_state = 2020,
                                        subsample = params['subsample'],
                                        tree_method = 'hist')
        xg_boost_model.fit(X_train, y_train)
        preds = xg_boost_model.predict(X_valid)
        mape = mean_absolute_percentage_error(y_valid, preds)
        print('mape', mape)
        return mape

    trials = Trials()
    best_hyperparams = optimize()
    hyperparameters = space_eval(space, best_hyperparams)

    #print best hyperparameters
    print('colsample_bytree', hyperparameters['colsample_bytree'])
    print('eta', hyperparameters['eta'])
    print('max_depth', hyperparameters['max_depth'])
    print('min_child_weight', hyperparameters['min_child_weight'])
    print('n_estimators', hyperparameters['n_estimators'])
    print('subsample', hyperparameters['subsample'])
    #xgb_model = XGBClassifier(hyperparameters)
    xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror',
                                colsample_bytree = hyperparameters['colsample_bytree'],
                                learning_rate = hyperparameters['eta'],
                                max_depth = hyperparameters['max_depth'],
                                min_child_weight = hyperparameters['min_child_weight'],
                                n_estimators = hyperparameters['n_estimators'],
                                random_state = 2020,
                                subsample = hyperparameters['subsample'],
                                tree_method = 'hist')
    xgb_model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
    

    """
    fname = './new_results/' + str(test_end.date()) + 'regress_12weeks_xgb_{}.csv'.format(args.state_name)
    test_df.to_csv(fname)

    print('test_metrics_{} '.format(args.state_name), test_metrics)
    json_result = json.dumps(test_metrics)
    with open('./new_results/' + str(test_end.date()) + 'regress_12weeks_xgb_{}.json'.format(args.state_name), 'w') as json_file:
        json_file.write(json_result)   
    pickle.dump(xgb_model, open('./xgboost_models/' + str(test_end.date()) + '{}_xgboost.pickle.dat'.format(args.state_name), "wb")) 
    """
    xgb_model.save_model('xgboost_models_1017/' + store_state[int(store_id)] + '-' + str(store_id) + '-' + 'xgboost.json')
    print(store_state[int(store_id)] + '-' + str(store_id) + '-' + ' XGBoost finished')