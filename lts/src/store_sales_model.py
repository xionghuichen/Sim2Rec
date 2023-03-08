"""
Store Sales Model to describle the relationship between the product price and the corresponding
sales in a week.

This model is developed based on Xgboost and Dunnhumby Dataset(https://www.dunnhumby.com/source-files/)

Author:Bowei He, Aug. 2022

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xgboost as xgb
from absl import flags
from absl import logging
import gin.tf
from gym import spaces
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

class StoreSalesModel:
    def __init__(self, store_id, merged_data, store_state, seed=0):
        self.model_path = '/home/ynmao/sim_rec_tf1/dunnhumby/xgboost_models_1017/'
        self.store_id = store_id
        #self.feature_list = ['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'STORE_NUM', 'UPC', 'PRICE', 'BASE_PRICE', 'STORE_ID', 'MSA_CODE', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']
        self.feature_list = ['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE']
        self.dataset = merged_data[merged_data['STORE_ID']==int(self.store_id)][self.feature_list + ['UNITS','UPC']].copy()
        self.product_list = self.dataset['UPC'].unique()
        self.prediction_model = xgb.Booster()
        self.prediction_model.load_model(self.model_path + store_state[store_id] + '-' + str(store_id) + '-' + 'xgboost.json')
        self.base_price = dict()
        price_mean = []
        price_std = []
        price = []
        self.mean_feature = dict()
        for product in self.product_list:
            self.base_price[product] = 0.0
            mean = self.dataset[self.dataset['UPC'] == product]['PRICE'].values.mean()
            std = self.dataset[self.dataset['UPC'] == product]['PRICE'].values.std()
            data = self.dataset[self.dataset['UPC'] == product]['PRICE'].values
            price_mean.append(mean)
            price_std.append(std)
            price.append(data)
            self.mean_feature[product] = self.dataset[self.dataset['UPC'] == product][['FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE']].values.mean(axis=0)
            #print('self.mean_feature[product].shape', self.mean_feature[product].shape)
        self.price_mean = np.array(price_mean)
        self.price_std = np.array(price_std)
        self.price = np.array(price)

            

    def predict_sales(self, price, time, product_id):
        # use product mean features to replace the 0.0 setting.
        model_input = self.dataset[(self.dataset['WEEK_END_DATE'] == str(time)) & (self.dataset['UPC'] == product_id)][self.feature_list]
        if model_input.empty:
            #model_input.loc[len(model_input.index)] = [0.0] * len(self.feature_list)
            model_input.loc[len(model_input.index)] = np.concatenate((np.array([0.0]), self.mean_feature[product_id]))
        model_input.loc[:,'PRICE'] = price
        model_input.loc[:,'BASE_PRICE'] = self.base_price[product_id]
        model_input.pop('WEEK_END_DATE') 
        predicted_sales = self.prediction_model.predict(xgb.DMatrix(model_input))

        product_obs = self.dataset[(self.dataset['WEEK_END_DATE'] == str(time + timedelta(weeks=1))) & (self.dataset['UPC'] == product_id)][self.feature_list]
        if product_obs.empty:
            #product_obs.loc[len(model_input.index)] = [0.0] * len(self.feature_list)
            product_obs.loc[len(model_input.index)] = np.concatenate((np.array([0.0]), self.mean_feature[product_id]))
        product_obs.pop('WEEK_END_DATE')
        product_obs.loc[:,'BASE_PRICE'] = price
        self.base_price[product_id] = price

        return np.array(predicted_sales), np.array(product_obs).squeeze()
