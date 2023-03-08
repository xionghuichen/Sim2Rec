from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import itertools

import six
from datetime import datetime, timedelta, date
import numpy as np
import random
"""
Class to represent the environmrnt in the intelligent pricing setting
Bowei He, Aug. 2022
"""
class SingleStoreEnvironment:
    def __init__(self, store_id, store_sales_model):
        self.store_id = store_id
        self.store_sales_model = store_sales_model
        self.len_episode = 4
        self.price_mean = store_sales_model.price_mean
        self.price_std = store_sales_model.price_std
        self.price = store_sales_model.price

        # prepare real trajectory sampling
        self.real_world_dataset =  self.store_sales_model.dataset[['WEEK_END_DATE', 'FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE', 'UNITS', 'UPC']]
        self.real_mean_feature = dict()
        for product in self.product_list:
            self.real_mean_feature[product] = self.real_world_dataset[self.real_world_dataset['UPC'] == product][['FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE', 'UNITS']].values.mean(axis=0)
    
    def reset(self):
        self.time = datetime.date(datetime.strptime('2009-01-14', '%Y-%m-%d')) + timedelta(weeks=random.randrange(0, 151 , 1))
        self.count = 0
        reset_all_product_obs = []
        reset_all_product_sales = []
        for product in self.product_list:
            reset_product_obs = self.store_sales_model.dataset[(self.store_sales_model.dataset['WEEK_END_DATE'] == str(self.time)) & (self.store_sales_model.dataset['UPC'] == product)][self.store_sales_model.feature_list[1:]]
            #reset_product_obs.pop('WEEK_END_DATE')
            if reset_product_obs.empty:
                reset_all_product_obs.append(np.zeros((len(self.store_sales_model.feature_list)-1)))
            else:
                reset_all_product_obs.append(reset_product_obs.values.squeeze())
            reset_all_product_sales.append(np.array([0.0]))
        print('reset_all_product_obs', reset_all_product_obs)
        print('np.array(reset_all_product_obs)', np.array(reset_all_product_obs))
        return np.array(reset_all_product_obs), np.array(reset_all_product_sales)

    @property
    def num_product(self):
        return len(self.store_sales_model.product_list)

    @property
    def product_list(self):
        return self.store_sales_model.product_list



    def step(self, prices):
        all_product_sales = []
        all_product_obs = []
        for i, product in enumerate(self.product_list):
            predicted_sales, product_obs = self.store_sales_model.predict_sales(prices[i], self.time, product)
            all_product_sales.append(predicted_sales)
            all_product_obs.append(product_obs)
        all_product_sales = np.array(all_product_sales)
        all_product_obs = np.array(all_product_obs)
        self.time = self.time + timedelta(weeks=1)
        self.count += 1
        if self.count > self.len_episode - 1:
            done = True
            self.reset()
        else:
            done = False
        return (all_product_obs, all_product_sales, done)

    def simulate_trajectory(self):
        trajectory_all_product_sales = []
        trajectory_all_product_obs = []
        self.reset()
        for i in range(self.len_episode):
            #prices = (self.price_max - self.price_min) * np.random.rand(self.num_product) + self.price_min
            prices = np.random.normal(self.price_mean, self.price_std, self.num_product)
            all_product_obs, all_product_sales, done = self.step(prices)
            trajectory_all_product_sales.append(all_product_sales)
            trajectory_all_product_obs.append(all_product_obs)
        trajectory_all_product_sales = np.array(trajectory_all_product_sales)
        trajectory_all_product_obs = np.array(trajectory_all_product_obs)
        trajectory = np.concatenate([trajectory_all_product_obs, trajectory_all_product_sales], axis=2)

        return trajectory

    def real_word_trajectory(self):
        trajectory = []   
        self.reset()
        for i in range(self.len_episode):
            step = []
            for j, product in enumerate(self.product_list):
                product_step = self.real_world_dataset[(self.real_world_dataset['WEEK_END_DATE'] == str(self.time)) & (self.real_world_dataset['UPC'] == product)][['FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE', 'UNITS']].values.squeeze()
                if product_step.size == 0:
                    product_step = self.real_mean_feature[product]
                step.append(product_step)   
            trajectory.append(np.array(step))
            self.time = self.time + timedelta(weeks=1)
        trajectory = np.array(trajectory)
        print('trajectory.shape', trajectory.shape)
        return trajectory

