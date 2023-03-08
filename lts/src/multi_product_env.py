import sys
sys.path.append('../../')

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats
import gin.tf
from common import logger
from lts.src.config import *
import pandas as pd 
import numpy as np
import json
from lts.src.store_sales_model import StoreSalesModel
from lts.src.dunnhumby_env import SingleStoreEnvironment
from lts.src import dunnhumby_gym 
from lts.src import dunnhumby_env

def make_dh(store_id, cgc_type=-1, log_sample=False):
    merged_data = pd.read_csv('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/merged_data_norm_1022.csv')
    with open('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/store_state.json') as f:
        store_state = json.load(f)
    
    store_model = StoreSalesModel(store_id, merged_data, store_state)
    store_env = dunnhumby_env.SingleStoreEnvironment(store_id, store_model)
    

    def selling_reward(sales):
        reward = 0
        for sale in sales:
            reward += sale
        return reward


    dh_gym_env = dunnhumby_gym.DunnhumbyGymEnvFlat(raw_environment=store_env, reward_aggregator=selling_reward)
    dh_gym_env_vec = dunnhumby_gym.DunnhumbyGymEnvVec(dh_gym_env, domain_name=store_id, cgc_type=cgc_type, log_sample=log_sample)

    return dh_gym_env_vec

if __name__ == '__main__':
    store_id = str(4259) #6379
    dunnhumby_lts_gym_env = make_dh(store_id)
    observation_0 = dunnhumby_lts_gym_env.reset()
    print('Observation 0 {}'.format(observation_0))

    action = np.ones(dunnhumby_lts_gym_env.env._environment.num_product)
    observation_1, reward, done, info = dunnhumby_lts_gym_env.step(action)
    print('Observation 1 {}'.format(observation_1))
    print("hidden state {}".format(info['hidden_state']))    



