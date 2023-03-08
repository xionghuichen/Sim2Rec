"""A wrapper for using Gym environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
from gym import spaces
import numpy as np
from common import logger
from lts.src.config import *
from stable_baselines.common.vec_env.base_vec_env import VecEnv

def _dh_metrics_aggregator(sales, metrics, info):
    return 

def _dh_metrics_writer(metrics, add_summary_fn):
    return 

class DunnhumbyGymEnv(gym.Env):
    def __init__(self, raw_environment, reward_aggregator, metrics_aggregator=_dh_metrics_aggregator, metrics_writer=_dh_metrics_writer):
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer
        self.reset_metrics()

    @property
    def environment(self):
        return self._environment
    @property
    def game_over(self):
        return False
    
    @property
    def action_space(self):
        action_space = spaces.Box(shape=(self._environment.num_product, 1), dtype=np.float32, low=0.0, high=np.inf)
        return action_space

    @property
    def observation_space(self):
        product_obs_space = spaces.Box(shape=(self._environment.num_product, len(self._environment.store_sales_model.feature_list)-1), dtype=np.float32, low=0.0, high=np.inf)
        produc_sales_space = spaces.Box(shape=(self._environment.num_product, 1), dtype=np.float32, low=0.0, high=np.inf)
        observation_space = spaces.Dict({
            'product_obs': product_obs_space,
            'product_sales': produc_sales_space,
        })
        return observation_space

    def step(self, action):
        all_product_obs, all_product_sales, done = self._environment.step(action)
        
        obs = dict(
            product_obs=all_product_obs,
            product_sales=all_product_sales
        )
        #extract reward
        reward = self._reward_aggregator(all_product_sales)
        info = self.extract_env_info()
        return obs, reward, done, info 
    
    def reset(self, *args, **kwargs):
        all_product_obs, all_product_sales = self._environment.reset()
        print('all_product_obs', all_product_obs)
        return dict(product_obs=all_product_obs, product_sales=all_product_sales)
    
    
    def seed(self, seed=None):
        np.random.seed(seed=seed)
    
    def extract_env_info(self):
        info = {'env': self._environment}
        return info

    def reset_metrics(self):
        self._metrics = collections.defaultdict(float)

    def update_metrics(self, all_product_sales, info=None):
        self._matrics = self._metrics_aggregator(all_product_sales, self._metrics, info)

    def write_metrics(self, add_summary_fn):
        self._metrics_writer(self._metrics, add_summary_fn)

    




class DunnhumbyGymEnvFlat(DunnhumbyGymEnv):
    def __init__(self, *args, **kwargs):
        super(DunnhumbyGymEnvFlat, self).__init__(*args, **kwargs)
        self.done = True
    
    @property
    def action_space(self):
        return super(DunnhumbyGymEnvFlat, self).action_space

    @property
    def observation_space(self):
        obs_space = super(DunnhumbyGymEnvFlat, self).observation_space
        lows = []
        highs = []
        for key, value in obs_space.spaces.items():
            print(key, ":", value)
            assert isinstance(value, spaces.Box)
            lows.append(value.low)
            highs.append(value.high)
        low = np.concatenate(lows, axis=1)
        high = np.concatenate(highs, axis=1)
        return spaces.Box(low=low, high=high)

    def _dict_to_flat_feature(self, obs_dict):
        if obs_dict['product_sales'] is None:
            obs_dict['product_sales'] = np.zeros((self._environment.num_product, 1))
        print('obs_dict[product_sales].shape', obs_dict['product_sales'].shape)
        print('obs_dict[product_obs].shape', obs_dict['product_obs'].shape)
        print('obs_dict[product_obs]', np.array(obs_dict['product_obs']).shape)
        print('obs_dict[product_obs]', np.array(obs_dict['product_obs']))
        merge_obs = np.concatenate([obs_dict['product_obs'], obs_dict['product_sales']], axis=1)
        return merge_obs


    #def reset(self, domain, evaluation):
    def reset(self, domain, evaluation):
        obs_dict = super(DunnhumbyGymEnvFlat, self).reset()
        return self._dict_to_flat_feature(obs_dict)

    def step(self, action):
        obs_dict, reward, done, info = super(DunnhumbyGymEnvFlat, self).step(action)
        self.done = done
        obs = self._dict_to_flat_feature(obs_dict)
        hidden_state = []
        for product in self._environment.product_list:
            hidden_state.append(None)
        info['hidden_state'] = np.array(hidden_state)
        return np.array(obs), np.array(reward), np.array([done]), info

    def need_reset(self, domain):
        return self.done

class DunnhumbyGymEnvVec(VecEnv):
    def __init__(self, env, domain_name, cgc_type=-1, log_sample=False):
        assert isinstance(env, DunnhumbyGymEnvFlat)
        self.env = env
        self.domain_list = [domain_name]
        self.cgc_type = cgc_type
        self.log_sample = log_sample
        self.time_budget = env._environment.len_episode
        self.domain = domain_name
        #super(DunnhumbyGymEnvVec, self).__init__(env._environment.num_product, env.observation_space, env.action_space)
        super(DunnhumbyGymEnvVec, self).__init__(1, env.observation_space, env.action_space)
        self.current_num_envs = self.num_envs


    def set_attr(self, attr_name, value, indices=None):
        return super().set_attr(attr_name, value, indices)
    
    def need_reset(self, domain):
        return self.env.need_reset(domain)

    def change_domain(self, domain_name):
        print('do nothing for change domain')

    def step_wait(self):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def close(self):
        raise NotImplementedError

    def step_async(self, actions):
        return super().step_async(actions)

    def get_attr(self, attr_name, indices=None):
        return super().get_attr(attr_name, indices)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)
    
    def clustering_hash(self, obs_list, mb_hidden_state):
        driver_cluster_hash = None
        return driver_cluster_hash

    def hash_to_array(self, obs_number, driver_cluster_hash):
        driver_cluster_list = None

        return driver_cluster_list


class MultiDomainGymEnv(VecEnv):
    def __init__(self, env_dict, domain_list, num_domain):
        self.env_dict = env_dict
        self.domain_list = domain_list
        self.num_domain = num_domain

        self.representive_env = self.env_dict[self.domain_list[0]]
        self.current_num_envs = self.representive_env.num_envs
        self.selected_env = self.representive_env

        assert isinstance(self.representive_env, DunnhumbyGymEnvVec)
        super(MultiDomainGymEnv, self).__init__(self.representive_env.num_envs, 
                                                self.representive_env.observation_space,
                                                self.representive_env.action_space)

    @property
    def domain(self):
        return self.selected_env.domain

    @property
    def time_budget(self):
        return self.selected_env.time_budget

    @property
    def hd_feature(self):
        return self.selected_env.hd_feature

    def change_domain(self, domain_name):
        self.selected_env = self.env_dict[domain_name]
        self.current_num_envs = self.selected_env.num_envs
    
    def seed(self, seed = None):
        return self.selected_env.seed(seed)

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def need_reset(self, domain):
        return self.selected_env.need_reset(domain)

    def step_wait(self):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def reset(self, domain, evaluation, *args, **kwargs):
        self.change_domain(domain)
        return self.selected_env.reset(domain, evaluation, *args, **kwargs)

    def step(self, action, *args, **kwargs):
        return self.selected_env.step(action, *args, **kwargs)

    def clustering_hash(self, obs_list, mb_hidden_state):
        return self.selected_env.clustering_hash(obs_list, mb_hidden_state)

    def hash_to_array(self, obs_number, driver_cluster_hash):
        return  self.selected_env.hash_to_array(obs_number, driver_cluster_hash)
