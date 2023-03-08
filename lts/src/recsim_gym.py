# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A wrapper for using Gym environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
from gym import spaces
import numpy as np
from lts.src import environment
from common import logger
from lts.src.config import *

def _dummy_metrics_aggregator(responses, metrics, info):
    del responses  # Unused.
    del metrics  # Unused.
    del info  # Unused.
    return


def _dummy_metrics_writer(metrics, add_summary_fn):
    del metrics  # Unused.
    del add_summary_fn  # Unused.
    return


class RecSimGymEnv(gym.Env):
    """Class to wrap recommender system environment to gym.Env.

    Attributes:
      game_over: A boolean indicating whether the current game has finished
      action_space: A gym.spaces object that specifies the space for possible
        actions.
      observation_space: A gym.spaces object that specifies the space for possible
        observations.
    """

    def __init__(self,
                 raw_environment,
                 reward_aggregator,
                 metrics_aggregator=_dummy_metrics_aggregator,
                 metrics_writer=_dummy_metrics_writer):
        """Initializes a RecSim environment conforming to gym.Env.

        Args:
          raw_environment: A recsim recommender system environment.
          reward_aggregator: A function mapping a list of responses to a number.
          metrics_aggregator: A function aggregating metrics over all steps given
            responses and response_names.
          metrics_writer:  A function writing final metrics to TensorBoard.
        """
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer
        self.reset_metrics()

    @property
    def environment(self):
        """Returns the recsim recommender system environment."""
        return self._environment

    @property
    def game_over(self):
        return False

    @property
    def action_space(self):
        """Returns the action space of the environment.

        Each action is a vector that specified document slate. Each element in the
        vector corresponds to the index of the document in the candidate set.
        """
        action_space = spaces.MultiDiscrete(
            self._environment.num_candidates * np.ones(
                (self._environment.slate_size,)
            ))
        if isinstance(self._environment, environment.MultiUserEnvironment):
            action_space = spaces.Tuple([action_space] * self._environment.num_users)
        return action_space

    @property
    def observation_space(self):
        """Returns the observation space of the environment.

        Each observation is a dictionary with three keys `user`, `doc` and
        `response` that includes observation about user state, document and user
        response, respectively.
        """
        if isinstance(self._environment, environment.MultiUserEnvironment):
            user_obs_space = self._environment.user_model[0].observation_space()
            resp_obs_space = self._environment.user_model[0].response_space()
            user_obs_space = spaces.Tuple(
                [user_obs_space] * self._environment.num_users)
            resp_obs_space = spaces.Tuple(
                [resp_obs_space] * self._environment.num_users)

        if isinstance(self._environment, environment.SingleUserEnvironment):
            user_obs_space = self._environment.user_model.observation_space()
            resp_obs_space = self._environment.user_model.response_space()

        return spaces.Dict({
            'user': user_obs_space,
            'doc': self._environment.candidate_set.observation_space(),
            'response': resp_obs_space,
        })

    def step(self, action):
        """Runs one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
          action (object): An action provided by the environment

        Returns:
          A four-tuple of (observation, reward, done, info) where:
            observation (object): agent's observation that include
              1. User's state features
              2. Document's observation
              3. Observation about user's slate responses.
            reward (float) : The amount of reward returned after previous action
            done (boolean): Whether the episode has ended, in which case further
              step() calls will return undefined results
            info (dict): Contains responses for the full slate for
              debugging/learning.
        """
        user_obs, doc_obs, responses, done = self._environment.step(action)
        if isinstance(self._environment, environment.MultiUserEnvironment):
            all_responses = tuple(
                tuple(
                    response.create_observation() for response in single_user_resps
                ) for single_user_resps in responses
            )
        else:  # single user environment
            all_responses = tuple(
                response.create_observation() for response in responses
            )
        obs = dict(
            user=user_obs,
            doc=doc_obs,
            response=all_responses)

        # extract rewards from responses
        reward = self._reward_aggregator(responses)
        info = self.extract_env_info()
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        user_obs, doc_obs = self._environment.reset()
        return dict(user=user_obs, doc=doc_obs, response=None)

    def reset_sampler(self):
        self._environment.reset_sampler()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=None):
        np.random.seed(seed=seed)

    def extract_env_info(self):
        info = {'env': self._environment}
        return info

    def reset_metrics(self):
        """Resets every metric to zero.

        We reset metrics for every iteration but not every episode. On the other
        hand, reset() gets called for every episode.
        """
        self._metrics = collections.defaultdict(float)

    def update_metrics(self, responses, info=None):
        """Updates metrics with one step responses."""
        self._metrics = self._metrics_aggregator(responses, self._metrics, info)

    def write_metrics(self, add_summary_fn):
        """Writes metrics to TensorBoard by calling add_summary_fn."""
        self._metrics_writer(self._metrics, add_summary_fn)


from stable_baselines.common.vec_env.base_vec_env import VecEnv


class RecSimGymEnvFlat(RecSimGymEnv):
    def __init__(self, *args, **kwargs):
        super(RecSimGymEnvFlat, self).__init__(*args, **kwargs)
    @property
    def action_space(self):
        acs_space = super(RecSimGymEnvFlat, self).action_space
        assert self._environment.slate_size == 1
        identity = np.ones(shape=(1,))
        return gym.spaces.Box(low=identity * -1, high=identity, dtype=np.float32)

    @property
    def observation_space(self):
        obs_space = super(RecSimGymEnvFlat, self).observation_space
        """
       return spaces.Dict({
            'user': user_obs_space,
            'doc': self._environment.candidate_set.observation_space(),
            'response': resp_obs_space,
        })
        """
        lows = []
        highs = []
        for key, value in obs_space.spaces.items():
            if key == 'doc':
                continue
            elif key == 'response':
                assert self._environment.slate_size == 1
                rep_space = value[0][0]
            elif key == 'user':
                rep_space = value[0]
            else:
                raise NotImplementedError
            print(key, ":", rep_space)
            assert isinstance(rep_space, spaces.Box)
            lows.append(rep_space.low)
            highs.append(rep_space.high)
        low = np.concatenate(lows)
        high = np.concatenate(highs)
        return spaces.Box(low=low, high=high)

    def _dict_to_flat_feature(self, obs_dict):
        if obs_dict['response'] is None:
            obs_dict['response'] = np.zeros((self._environment.num_users, self._environment.slate_size, self._environment.user_model[0].response_space()[0].shape[0]))
        merge_obs = []
        for user_obs, res_obs in zip(obs_dict['user'], obs_dict['response']):
            assert self._environment.slate_size == 1
            res_obs = res_obs[0]
            merge_obs.append(np.concatenate([user_obs, res_obs]))
        merge_obs = np.array(merge_obs)
        return merge_obs

    def reset(self, domain, evaluation):
        obs_dict = super(RecSimGymEnvFlat, self).reset()
        return self._dict_to_flat_feature(obs_dict)

    def step(self, action):
        action = (((action + 1) / 2) * (self._environment.num_candidates - 1)).astype(np.int32)
        obs_dict, reward, done, info = super(RecSimGymEnvFlat, self).step(action)
        obs = self._dict_to_flat_feature(obs_dict)
        hidden_state = []
        for um in self._environment.user_model:
            hidden_state.append(um._user_state.create_hidden_state())
        info['hidden_state'] = np.array(hidden_state)
        done = [user_model.is_terminal() for user_model in self._environment.user_model]

        return np.array(obs), np.array(reward), np.array(done), info

    def need_reset(self, domain):
        return all([user_model.is_terminal() for user_model in self._environment.user_model])


class RecSimGymEnvVec(VecEnv):
    def __init__(self, env, domain_name, cgc_type=-1, log_sample=False):
        assert isinstance(env, RecSimGymEnvFlat)
        self.env = env
        self.domain_list = [domain_name]
        self.cgc_type = cgc_type
        self.log_sample = log_sample

        if self.cgc_type > 0:
            if not log_sample:
                self.satisfication_bin_edge = np.arange(SATISFACTION_RANGE[0], SATISFACTION_RANGE[1], (SATISFACTION_RANGE[1] - SATISFACTION_RANGE[0]) / self.cgc_type)
                self.memory_discount_bin_edge = np.arange(MEMORY_DISCOUNT_RANGE[0], MEMORY_DISCOUNT_RANGE[1], (MEMORY_DISCOUNT_RANGE[1] - MEMORY_DISCOUNT_RANGE[0]) / self.cgc_type)
                self.sensitivity_bin_edge = np.arange(SENSITIVITY_RANGE[0], SENSITIVITY_RANGE[1], (SENSITIVITY_RANGE[1] - SENSITIVITY_RANGE[0]) / self.cgc_type)
            else:
                self.satisfication_bin_edge = np.arange(SATISFACTION_RANGE[0], SATISFACTION_RANGE[1], (SATISFACTION_RANGE[1] - SATISFACTION_RANGE[0]) / self.cgc_type)
                self.memory_discount_bin_edge = np.arange(LOG_MEMORY_DISCOUNT_RANGE[0], LOG_MEMORY_DISCOUNT_RANGE[1], (LOG_MEMORY_DISCOUNT_RANGE[1] - LOG_MEMORY_DISCOUNT_RANGE[0]) / self.cgc_type)
                self.sensitivity_bin_edge = np.arange(LOG_SENSITIVITY_RANGE[0], LOG_SENSITIVITY_RANGE[1],( LOG_SENSITIVITY_RANGE[1] - LOG_SENSITIVITY_RANGE[0]) / self.cgc_type)
        self.domain = domain_name
        self.time_budget = env._environment.user_model[0]._user_state.time_budget
        self.hd_feature = env._environment.user_model[0]._user_state.hd_list
        super(RecSimGymEnvVec, self).__init__(1, env.observation_space, env.action_space)
        self.current_num_envs = self.num_envs
    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def need_reset(self, domain):
        return self.env.need_reset(domain)

    def change_domain(self, domain_name):
        print("do nothing for change domain")

    def step_wait(self):
        raise NotImplementedError
    
    def seed(self, seed = None):
        return self.env.seed(seed)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def clustering_hash(self, obs_list, mb_hidden_state):
        satisfication = mb_hidden_state[:, self.hd_feature.index('satisfication')]
        if not self.log_sample:
            memory_discount = mb_hidden_state[:, self.hd_feature.index('memory_discount')]
            sensitivity = mb_hidden_state[:, self.hd_feature.index('sensitivity')]
        else:
            memory_discount = np.exp(mb_hidden_state[:, self.hd_feature.index('memory_discount')])
            sensitivity = np.exp(mb_hidden_state[:, self.hd_feature.index('sensitivity')])
        driver_cluster_hash = {}
        stais_class_type = np.digitize(satisfication, self.satisfication_bin_edge)
        memory_discount_class_type = np.digitize(memory_discount, self.memory_discount_bin_edge)
        sensitivity_class_type = np.digitize(sensitivity, self.sensitivity_bin_edge)

        # for i in range(0, self.satisfication_bin_edge.shape[0] + 1):
        for j in range(0, self.memory_discount_bin_edge.shape[0] + 1):
            for k in range(0, self.sensitivity_bin_edge.shape[0] + 1):
                #  (stais_class_type == i) &
                idx = np.where((memory_discount_class_type == j) & (sensitivity_class_type == k))[0]
                if idx.shape[0] > 0:
                    driver_cluster_hash['{}-{}'.format(j, k)] = idx
        return driver_cluster_hash

    def hash_to_array(self, obs_number, driver_cluster_hash):
        small_cluster_count = 0
        cluster_number = len(driver_cluster_hash.keys())
        driver_cluster_list = np.ones(shape=(cluster_number, obs_number), dtype=np.int32) * -1
        index = 0
        for k, v in driver_cluster_hash.items():
            if len(v) <= 3:
                small_cluster_count += 1
            driver_cluster_list[index][:len(v)] = v
            index += 1
        logger.record_tabular("env-{}/cluster_small_than_3".format(self.domain), small_cluster_count / cluster_number)
        logger.record_tabular("env-{}/clusters".format(self.domain), len(driver_cluster_hash.keys()))
        return driver_cluster_list

class MultiDomainGymEnv(VecEnv):

    def __init__(self, env_dict, domain_list, num_domain):
        self.env_dict = env_dict
        self.domain_list = domain_list
        self.num_domain = num_domain

        self.representive_env = self.env_dict[self.domain_list[0]]
        self.current_num_envs = self.representive_env.num_envs
        self.selected_env = self.representive_env
        assert isinstance(self.representive_env, RecSimGymEnvVec)
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
