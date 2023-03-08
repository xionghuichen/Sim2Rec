'''
Environment for simulating driver state-action transition.
'''
import matplotlib

import gym
from gym import spaces
from gym.utils import seeding
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import random
import os
from common import logger
import pickle
from common.utils import *
from common.tester import tester
from common.config import *
from common.delay_buffer import DelayBuffer
import baselines.common.tf_util as U
from common.data_info import CouponActionInfo
from common.statistic import Statistic
from common.env_base import BaseDriverEnv
import gym
from stable_baselines.common.vec_env.base_vec_env import VecEnv

from common.emb_unit_test import emb_unit_test_cont
from common.mpi_running_mean_std import RunningMeanStdNp

class MultiCityDriverEnv(VecEnv):

    def __init__(self, city_env_dict, city_list, representation_city_name, cp_unit_mask):
        self.city_env_dict = city_env_dict
        self.city_list = city_list
        self.representative_city = self.city_env_dict[representation_city_name]
        # self.action_scale_coeff = action_scale_coeff
        self.cp_unit_mask = cp_unit_mask
        assert isinstance(self.representative_city, DriverEnv)
        num_envs = int(np.round(len(self.representative_city.driver_ids) * self.representative_city.sample_percent))
        self.hand_code_driver_action = self.representative_city.hand_code_driver_action
        # set env space information.
        if self.cp_unit_mask:
            self.reveal_acs_dim = 1
        else:
            self.reveal_acs_dim = 2
        identity = np.ones(shape=(self.representative_city.dim_state, ))
        low = -1 * np.inf * identity
        high = np.inf * identity
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32, )
        identity = np.ones(shape=(self.reveal_acs_dim, ))
        low = -1 * np.inf * identity
        high = np.inf * identity
        action_space = gym.spaces.Box(low=identity * -1, high=identity, dtype=np.float32)
        super(MultiCityDriverEnv, self).__init__(num_envs, observation_space, action_space)
        self.current_num_envs = num_envs
        self.selected_env = None
        self.selected_city = self.representative_city
        self.evaluation_type = EvaluationType.NO
        self.multi_ts = {}
        self.multi_epi_info = {}

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

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

    @property
    def driver_action_space(self):
        return gym.spaces.Box(low=np.array([-1 * np.inf]), high=np.array([np.inf]), dtype=np.float32, )

    @property
    def days(self):
        return self.representative_city.T - self.representative_city.delay_buffer.max_delay

    @property
    def city_len(self):
        return len(self.city_list)

    @property
    def feature_name(self):
        # TODO: patch for vae dimension error.
        feature_name = np.array(self.representative_city.features_set)
        # dim_dac_cac = self.representative_city.dim_coupon_feature + self.representative_city.dim_action
        # feature_name = np.concatenate([feature_name[:self.representative_city.dim_state], feature_name[-1 * self.representative_city.dim_coupon_feature:-1 * self.representative_city.dim_action]])
        return feature_name

    def extractor_dacs(self, obs):
        acs = obs[:, self.representative_city.dim_static_feature:]
        restore_fos = self.selected_env.compute_fos(acs)
        if len(restore_fos.shape) == 1:
            restore_fos = np.expand_dims(restore_fos, axis=-1).astype('float32')
        return restore_fos

    def rescale_action(self, acs, env):
        res_acs = (acs + 0.9) / 2 * (env.cac_max_value - env.cac_min_value) * 0.9 + env.cac_min_value
        return res_acs

    def restore_action(self, acs):
        if acs is None:
            return None
        clipped_actions = np.clip(acs, self.action_space.low, self.action_space.high)
        clipped_actions = np.concatenate([clipped_actions, np.zeros(shape=[clipped_actions.shape[0], 5 - self.reveal_acs_dim])], axis=1)
        res_acs = self.rescale_action(clipped_actions, self.representative_city)
        return self.representative_city.restore_coupon_info(res_acs, reshape_sec_fos=False)

    def change_city(self, city):
        self.selected_city = city
        self.selected_env = self.city_env_dict[self.selected_city]
        if city not in self.multi_epi_info:
            self.multi_epi_info[self.selected_city] = self.init_epi_info()
            self.multi_ts[self.selected_city] = 0
        self.epi_info = self.multi_epi_info[self.selected_city]
        self.ts = self.multi_ts[self.selected_city]
        self.current_num_envs = self.selected_env.driver_number

    def _merge_concat(self, inp):
        return np.concatenate(inp, axis=0)

    def merge_reset(self, *args, **kwargs):
        observation = []
        for city in self.city_list:
            self.change_city(city)
            obs = self.reset(city=city)
            observation.append(obs)
        observation = self._merge_concat(observation)
        return observation


    def reset(self, evaluation=EvaluationType.NO, city=None, *args, **kwargs):
        if city is None:
            city_id = np.random.randint(0, self.city_len)
            self.selected_city = self.city_list[city_id]
        else:
            self.selected_city = city
        # if self.need_reset[self.selected_city]:
        #     self.need_reset[self.selected_city] = False
        # else:
        #     raise Exception

        self.change_city(self.selected_city)
        assert isinstance(self.selected_env, DriverEnv)
        if evaluation == EvaluationType.NO:
            obs = self.selected_env.reset(scaled_budget=True, *args, **kwargs)
        else:
            obs = self.selected_env.reset(percent=1, scaled_budget=False, *args, **kwargs)
        self.evaluation_type = evaluation
        self.current_num_envs = self.selected_env.driver_number
        self.multi_epi_info[self.selected_city] = self.epi_info = self.init_epi_info()
        self.multi_ts[self.selected_city] = self.ts = 0
        return obs
    
    def need_reset(self, city=None):
        if city is None:
            city = self.selected_city
        return self.city_env_dict[city].need_reset

    def reshape_and_step(self, flat_fo_pred):
        slice_start = 0
        assert len(self.city_list) == 1, "多测试城市，此处需要设计merge操作"
        city_env = self.selected_env
        cp_unit_cost_number, cp_bound_number = city_env.gen_action_info()
        slice_end = slice_start + cp_unit_cost_number * cp_bound_number * city_env.driver_number
        assert flat_fo_pred.shape[0] == slice_end
        stand_cp = city_env.reshape_and_gen_action(flat_fo_pred[slice_start:slice_end])
        return self.step(stand_cp[:, :self.reveal_acs_dim])

    def merge_step(self, action=None, *args, **kwargs):
        acs_idx_s = 0
        observations = []
        rewards = []
        batch_dones = []
        infos = []
        for city in self.city_list:
            self.change_city(city)
            slice_acs = action[acs_idx_s: acs_idx_s + self.selected_env.driver_number]
            obs, reward, batch_done, info = self.step(slice_acs, *args, **kwargs)
            acs_idx_s += self.selected_env.driver_number

            rewards.append(reward)
            batch_dones.append(batch_done)

            if self.need_reset():
                obs = self.reset()
                infos.append([info['episode']])
            else:
                infos.append([])
            observations.append(obs)
        return map(self._merge_concat, (observations, rewards, batch_dones, infos))


    def step(self, action=None, *args, **kwargs):
        if action is not None:
            action = self.restore_action(action)
            if self.cp_unit_mask:
                action[:, 1] = self.selected_env.cac_mean_std[1, 0]
            action[:, 2:] = [0, 0, -999] # self.selected_env.cac_mean_std[2:, 0]
            rescale_action = self.selected_env.cp_info.construct_coupon_info(action, sec_reshape=False, is_r_factor=True)
        else:
            rescale_action = None
        cost_constraint = self.evaluation_type == EvaluationType.COST_CONSTRAINT
        obs, reward, batch_done, info = self.selected_env.step(rescale_action, cost_constraint=cost_constraint, *args, **kwargs)
        self.append_epi_info(reward, info)
        # self.update_need_reset(batch_done)
        if self.need_reset():
            info = self.push_epi_info(info)
        self.multi_ts[self.selected_city] += 1
        self.ts += 1
        return obs, reward, batch_done, info

    def demer_mean_prop_test(self):
        logger.info("demer_mean_prop_test")
        for city in self.city_list:
            self.city_env_dict[city].demer_mean_prop_test()

    def demer_data_gen(self, *args, **kwargs):
        dfs = []
        for city in self.city_list:
            df, feature_name_path = self.city_env_dict[city].demer_data_gen(*args, **kwargs)
            dfs.append(df)
        return pd.concat(dfs), feature_name_path

    def gen_all_sa(self, obs):
        dfs = []
        for city in self.city_list:
            df = self.city_env_dict[city].gen_all_sa(obs)
            dfs.append(df)
        return pd.concat(dfs)


    def demer_prop_test(self):
        logger.info("demer_prop_test")
        for city in self.city_list:
            self.city_env_dict[city].demer_prop_test()

    def expert_policy_info_gen(self, compared_env=None):
        assert compared_env is None or isinstance(compared_env, MultiCityDriverEnv)
        max_acs = self.restore_action(np.array([self.action_space.high]))
        for city in self.city_list:
            logger.info("expert_policy_info_gen : {}".format(city))
            if self.city_env_dict[city].expert_info is not None:
                logger.info("skip")
                continue
            if self.cp_unit_mask:
                self.city_env_dict[city].max_unit_cost = self.city_env_dict[city].cac_mean_std[1, 0]
            else:
                self.city_env_dict[city].max_unit_cost = max_acs[0, 1]
            logger.info("max unit cost {}".format(max_acs[0, 1]))
            if self.city_env_dict[city].expert_info is None:
                self.change_city(city)
                self.reset(evaluation=True, city=city, run_one_step=not self.hand_code_driver_action)
                info = {}
                for i in range(self.days):
                    _, _, _, info = self.step(dynamic_type=DynamicType.EXPERT)
                # emb the pic plot process.
                name = 'data_compare-{}'.format(city)
                sim_data = info['episode']['daily_gmv'].sum(axis=1) / self.selected_env.unit_gmv / 100
                real_data = info['episode']['daily_real_data_gmv'].sum(axis=1) / self.selected_env.unit_gmv / 100
                labels = ['real', 'sim']
                data = np.stack([real_data, sim_data], axis=0)
                tester.simple_plot(name, data, labels, pretty=True, xlabel='time-step (per day)', ylabel='scaled FOs',
                                    colors=['r', 'b'], styles=['x--', '+-'])
                sim_data_fos = info['episode']['daily_gmv'].flatten() / self.selected_env.unit_gmv / 100
                real_data_fos = info['episode']['daily_real_data_gmv'].flatten() / self.selected_env.unit_gmv / 100
                data_fos = np.stack([real_data_fos, sim_data_fos], axis=0)
                tester.simple_hist('fos-{}'.format(city), data_fos, labels, pretty=True, xlabel='scaled FOs', ylabel='density',
                                   bins=int(real_data_fos.max() * 100) - 2, density=True, log=True)
                assert self.need_reset(city)
                assert 'episode' in info
                self.selected_env.expert_info = info['episode']
                self.selected_env.coupon_info_bound = max_acs
                logger.record_tabular("target-{}/expert-cost".format(city), np.mean(self.selected_env.expert_info['cost']))
                logger.record_tabular("target-{}/expert-rews".format(city), np.mean(self.selected_env.expert_info['real_rews']))
                logger.record_tabular("target-{}/expert-gmv".format(city), np.mean(self.selected_env.expert_info['gmv']))
                logger.record_tabular("target-{}/expert-coupon_send".format(city), np.mean(self.selected_env.expert_info['coupon_send']))
                self.change_city(city)
                self.reset(evaluation=True, city=city, run_one_step=not self.hand_code_driver_action)
                info = {}
                for i in range(self.days):
                    _, _, _, info = self.step(dynamic_type=DynamicType.EXPERT_DATA)
                info = info['episode']
                logger.record_tabular("target-{}/expert-data-cost".format(city), np.mean(info['cost']))
                logger.record_tabular("target-{}/expert-data-rews".format(city), np.mean(info['real_rews']))
                logger.record_tabular("target-{}/expert-data-gmv".format(city), np.mean(info['gmv']))
                logger.record_tabular("target-{}/expert-data-coupon_reached".format(city), np.mean(info['coupon_reached']))
                logger.record_tabular("target-{}/expert-data-coupon_send".format(city), np.mean(info['coupon_send']))
                coupon1 = info['daily_coupon1'].flatten()
                coupon2 = info['daily_coupon2'].flatten()
                coupon1 = coupon1[np.where(coupon1 > 0)]
                coupon2 = coupon2[np.where(coupon2 > 0)]
                tester.simple_hist('{}/expert-data-coupon1'.format(city), data=coupon1, bins=int(np.max(coupon1)))
                tester.simple_hist('{}/expert-data-coupon2'.format(city), data=coupon2, bins=30)
                self.change_city(city)
                self.reset(evaluation=True, city=city)
                for i in range(self.days):
                    if (i == 0 or i == self.days - 1) and self.hand_code_driver_action:
                        random_scale = self.selected_env.get_random_sacle(obs_list=self.selected_env.state)
                        tester.simple_hist('random_scale-{}-{}'.format(city, i), random_scale.T,
                                           labels=['mean', 'std', 'max', 'operated', 'sensitive'], bins=100)
                    _, _, _, info = self.step(dynamic_type=DynamicType.OPTIMAL)
                    # coupon1 = info['coupon'][:, 0].flatten()
                    # coupon2 = info['coupon'][:, 1].flatten()
                    # tester.simple_hist('coupon1-{}-{}'.format(city, i), data=coupon1, bins=int(np.max(coupon1)))
                    # tester.simple_hist('coupon2-{}-{}'.format(city, i), data=coupon2, bins=30)
                assert self.need_reset(city)
                assert 'episode' in info
                self.selected_env.optimal_info = info['episode']
                logger.record_tabular("target-{}/optimal-cost".format(city), np.mean(self.selected_env.optimal_info['cost']))
                logger.record_tabular("target-{}/optimal-rews".format(city), np.mean(self.selected_env.optimal_info['real_rews']))
                logger.record_tabular("target-{}/optimal-gmv".format(city), np.mean(self.selected_env.optimal_info['gmv']))

                coupon1 = info['episode']['daily_coupon1'].flatten()
                coupon2 = info['episode']['daily_coupon2'].flatten()
                tester.simple_hist('{}/coupon1-rule'.format(city), data=coupon1, bins=int(np.max(coupon1)))
                tester.simple_hist('{}/coupon2-rule'.format(city), data=coupon2, bins=30)
                self.change_city(city)
                self.reset(evaluation=True, city=city)
                for i in range(self.days):
                    if (i == 0 or i == self.days - 1) and self.hand_code_driver_action:
                        random_scale = self.selected_env.get_random_sacle(obs_list=self.selected_env.state)
                        tester.simple_hist('random_scale-{}-{}'.format(city, i), random_scale.T,
                                           labels=['mean', 'std', 'max', 'operated', 'sensitive'], bins=100)
                    _, _, _, info = self.step(dynamic_type=DynamicType.SUB_OPTIMAL)
                    # coupon1 = info['coupon'][:, 0].flatten()
                    # coupon2 = info['coupon'][:, 1].flatten()
                    # tester.simple_hist('coupon1-{}-{}'.format(city, i), data=coupon1, bins=int(np.max(coupon1)))
                    # tester.simple_hist('coupon2-{}-{}'.format(city, i), data=coupon2, bins=30)
                assert self.need_reset(city)
                assert 'episode' in info
                self.selected_env.sub_optimal_info = info['episode']
                logger.record_tabular("target-{}/sub_optimal-cost".format(city),
                                      np.mean(self.selected_env.sub_optimal_info['cost']))
                logger.record_tabular("target-{}/sub_optimal-rews".format(city),
                                      np.mean(self.selected_env.sub_optimal_info['real_rews']))
                logger.record_tabular("target-{}/sub_optimal-gmv".format(city),
                                      np.mean(self.selected_env.sub_optimal_info['gmv']))

                coupon1 = info['episode']['daily_coupon1'].flatten()
                coupon2 = info['episode']['daily_coupon2'].flatten()
                tester.simple_hist('{}/coupon1-sub-rule'.format(city), data=coupon1, bins=int(np.max(coupon1)))
                tester.simple_hist('{}/coupon2-sub-rule'.format(city), data=coupon2, bins=30)

            opt_rew = np.mean(self.selected_env.optimal_info['real_rews'])

            if self.hand_code_driver_action and compared_env is not None:
                self.city_env_dict[city].replace_env_scaled_info_gen(self.city_env_dict[city[:-5]])
                self.change_city(city)
                self.reset(evaluation=True, city=city, unit_test_type=UnitTestType.SUB_OPTIMAL_TEST)
                diff_time_info = None
                for i in range(self.days):
                    _, _, _, diff_time_info = self.step(dynamic_type=DynamicType.SUB_OPTIMAL)
                self.selected_env.time_diff_optimal_info = diff_time_info['episode']
                logger.record_tabular("target-{}/time-diff-cost".format(city), np.mean(self.selected_env.time_diff_optimal_info['cost']))
                logger.record_tabular("target-{}/time-diff-rews".format(city), np.mean(self.selected_env.time_diff_optimal_info['real_rews']))
                logger.record_tabular("target-{}/time-diff-gmv".format(city), np.mean(self.selected_env.time_diff_optimal_info['gmv']))
                logger.record_tabular("target-{}/time-diff-rew_percent".format(city), np.mean(self.selected_env.time_diff_optimal_info['real_rews']) / opt_rew)
                best_info, best_rew = None, 0
                worst_info, worst_rew = None, 99999
                rew_percent_list = []
                for rep_city in compared_env.city_list:
                    if rep_city == city:
                        continue
                    self.city_env_dict[city].replace_env_scaled_info_gen(self.city_env_dict[rep_city])
                    self.change_city(city)
                    self.reset(evaluation=True, city=city, unit_test_type=UnitTestType.SUB_OPTIMAL_TEST)
                    diff_city_info = None
                    for i in range(self.days):
                        _, _, _, diff_city_info = self.step(dynamic_type=DynamicType.SUB_OPTIMAL)
                    rew = np.mean(diff_city_info['episode']['real_rews'])
                    if best_rew < rew:
                        best_info = diff_city_info
                        best_rew = rew
                    if worst_rew > rew:
                        worst_rew = rew
                        worst_info = diff_city_info
                    rew_percent_list.append(rew / opt_rew)
                self.selected_env.city_diff_best_info = best_info['episode']
                self.selected_env.city_diff_worst_info = worst_info['episode']
                logger.record_tabular("target-{}/city-change-best-cost".format(city), np.mean(self.selected_env.city_diff_best_info['cost']))
                logger.record_tabular("target-{}/city-change-best-rews".format(city), np.mean(self.selected_env.city_diff_best_info['real_rews']))
                logger.record_tabular("target-{}/city-change-best-gmv".format(city), np.mean(self.selected_env.city_diff_best_info['gmv']))
                logger.record_tabular("target-{}/city-change-best-rew_percent".format(city), np.mean(self.selected_env.city_diff_best_info['real_rews']) / opt_rew)

                logger.record_tabular("target-{}/city-change-worst-cost".format(city), np.mean(self.selected_env.city_diff_worst_info['cost']))
                logger.record_tabular("target-{}/city-change-worst-rews".format(city), np.mean(self.selected_env.city_diff_worst_info['real_rews']))
                logger.record_tabular("target-{}/city-change-worst-gmv".format(city), np.mean(self.selected_env.city_diff_worst_info['gmv']))
                logger.record_tabular("target-{}/city-change-worst-rew_percent".format(city), np.mean(self.selected_env.city_diff_worst_info['real_rews']) / opt_rew)
                logger.record_tabular("target/city-diff-rew-percent-mean", np.mean(rew_percent_list))
                logger.record_tabular("target/city-diff-rew-percent-std", np.std(rew_percent_list))
                # logger.dump_tabular()
            self.selected_env.make_expert_statistics()


    def cluster_driver(self, obs_list):
        return self.selected_env.clustering_driver(obs_list)

    def clustering_hash(self, obs_list):
        return self.selected_env.clustering_hash(obs_list)

    def hash_to_array(self, obs_number, hash):
        return self.selected_env.hash_to_array(obs_number, hash)

    def init_epi_info(self):
        return {
                         "daily_rews": np.zeros([self.days, self.current_num_envs]),
                         "daily_real_rews": np.zeros([self.days, self.current_num_envs]),
                         "daily_null_coupon_rews": np.zeros([self.days, self.current_num_envs]),
                         "daily_cost": np.zeros([self.days, self.current_num_envs]),
                         "daily_revenue": np.zeros([self.days, self.current_num_envs]),
                         "daily_gmv": np.zeros([self.days, self.current_num_envs]),
                         "daily_coupon_reached": np.zeros([self.days, self.current_num_envs]),
                         "daily_coupon_send": np.zeros([self.days, self.current_num_envs]),
                         "daily_kl": np.zeros([self.days, self.current_num_envs]),
                         "daily_ue_penalty": np.zeros([self.days, self.current_num_envs]),
                         "daily_budget_penalty": np.zeros([self.days, self.current_num_envs]),
                         "daily_real_data_gmv": np.zeros([self.days, self.current_num_envs]),
                         "daily_real_data_cost": np.zeros([self.days, self.current_num_envs]),
                         "daily_coupon1": np.zeros([self.days, self.current_num_envs]),
                         "daily_coupon2": np.zeros([self.days, self.current_num_envs]),}

    def append_epi_info(self, rew, info):
        self.epi_info['daily_rews'][self.ts] = rew
        self.epi_info['daily_real_rews'][self.ts] = info['real_rews']
        self.epi_info['daily_null_coupon_rews'][self.ts] = info['null_coupon_rews']
        self.epi_info['daily_cost'][self.ts] = info['cost']
        self.epi_info['daily_revenue'][self.ts] = info['revenue']
        self.epi_info['daily_coupon_reached'][self.ts] = info['coupon_reached']
        self.epi_info['daily_coupon_send'][self.ts] = info['coupon_send']
        self.epi_info['daily_kl'][self.ts] = info['kl']
        self.epi_info['daily_ue_penalty'][self.ts] = info['ue_penalty']
        self.epi_info['daily_budget_penalty'][self.ts] = info['budget_penalty']
        self.epi_info['daily_gmv'][self.ts] = info['gmv']
        self.epi_info['daily_real_data_gmv'][self.ts] = info['real'][Statistic.AVG_GMV]
        self.epi_info['daily_coupon1'][self.ts] = info['coupon'][:, 0]
        self.epi_info['daily_coupon2'][self.ts] = info['coupon'][:, 1]

    def push_epi_info(self, info):
        self.epi_info['rews'] = np.sum(self.epi_info['daily_rews'], axis=0)
        self.epi_info['real_rews'] = np.sum(self.epi_info['daily_real_rews'], axis=0)
        self.epi_info['null_coupon_rews'] = np.sum(self.epi_info['daily_null_coupon_rews'], axis=0)
        self.epi_info['cost'] = np.sum(self.epi_info['daily_cost'], axis=0)
        self.epi_info['kl'] = np.mean(self.epi_info['daily_kl'], axis=0)
        self.epi_info['ue_penalty'] = np.mean(self.epi_info['daily_ue_penalty'], axis=0)
        self.epi_info['budget_penalty'] = np.mean(self.epi_info['daily_budget_penalty'], axis=0)
        self.epi_info['revenue'] = np.mean(self.epi_info['daily_revenue'], axis=0)
        self.epi_info['coupon_reached'] = np.mean(self.epi_info['daily_coupon_reached'], axis=0)
        self.epi_info['coupon_send'] = np.mean(self.epi_info['daily_coupon_send'], axis=0)
        self.epi_info['gmv'] = np.sum(self.epi_info['daily_gmv'], axis=0)
        self.epi_info['real_data_gmv'] = np.sum(self.epi_info['daily_real_data_gmv'], axis=0)
        self.epi_info['real_data_cost'] = np.sum(self.epi_info['daily_real_data_cost'], axis=0)

        self.epi_info['inc_rew'] = self.epi_info['real_rews'] - self.epi_info['null_coupon_rews']
        self.epi_info['coupon_ratio'] = np.sum(np.clip(self.epi_info['cost'], 0, None)) / (np.sum(self.epi_info['real_rews']) + 0.1)
        self.epi_info['roi'] = np.sum(np.clip(np.sum(self.epi_info['inc_rew']), 0, None)) / (np.sum(self.epi_info['cost']) + 0.1)
        self.epi_info['city_name'] = self.selected_env.name
        info['episode'] = self.epi_info.copy()
        return info

class DriverEnv(BaseDriverEnv):
    def __init__(self, policy_fn, coupon_ratio_scale, adv_rew,
                 re_construct_ac, sim_version, load_model_path,
                 rew_scale, unit_cost_penlaty,  only_gmv, mdp_env,
                 deter_env, kl_penalty, cp_unit_mask, gmv_rescale, remove_hist_state,
                 *args, **kwargs):
        BaseDriverEnv.__init__(self,  *args, **kwargs)
        self.policy_fn = policy_fn
        self.cp_unit_mask = cp_unit_mask
        self.unit_cost_penlaty = unit_cost_penlaty
        self.remove_hist_state = remove_hist_state
        self.only_gmv = only_gmv
        self.mdp_env = mdp_env
        self.load_model_path = load_model_path
        self.adv_rew = adv_rew
        self.rew_scale = rew_scale
        self.coupon_ratio_scale = coupon_ratio_scale
        self.coupon_ratio = self.coupon_ratio * coupon_ratio_scale
        self.unit_gmv = 20.0 # float(self.dac_mean_std[4, 0] / self.dac_mean_std[0, 0])
        logger.info("[WARNING] use default unit GMV : {}".format(self.unit_gmv))
        self.re_construct_ac = re_construct_ac
        # assert self.sample_percent > 0.2
        self.sim_version = sim_version
        self.deter_env = deter_env
        self.env_z_info = None
        self.expert_info = None
        self.optimal_info = None
        self.coupon_info_bound = None
        self.dynamic_type = None
        self.max_unit_cost = None

        self.gmv_rescale = gmv_rescale
        self.kl_penalty = kl_penalty
        from collections import deque
        self.rew_history = RunningMeanStdNp()
        self.kl_history = RunningMeanStdNp()
        self.ue_penalty_history = RunningMeanStdNp()
        self.budget_history = RunningMeanStdNp()

    def demer_mean_prop_test(self):
        standard_mean = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
        fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
        mean_fos = np.clip((standard_mean * fostd + fomean).astype("int"), 0, None)
        for day in range(self.delay_buffer.max_delay, self.T, 5):
            not_delay_state = self.traj_real_obs[day].copy()
            # delay_state = self.traj_real_obs[day - self.delay_buffer.max_delay]
            multi_coupon_react_ratios = []
            unit_coupon = self.cac_mean_std[1, 0]
            react_ratios = []
            coupon_fos = []
            # probs = []
            for i in range(1, 50):
                stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                coupon_action[:, ] = stand_cp
                # res = self.pi.cact_prob(delay_state)
                # policy_mean, policy_std = res[0][0], res[0][1]
                # policy_logstd = np.log(policy_std)
                # neg_log_prob = 0.5 * np.sum(np.square((stand_cp - policy_mean) / policy_std), axis=-1) \
                #                + 0.5 * np.log(2.0 * np.pi) * stand_cp.shape[-1] \
                #                + np.sum(policy_logstd, axis=-1)
                # prob = np.exp(-1 * neg_log_prob)
                inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                dac, value = self.pi.act(False, inner_state, coupon_action,
                                         self.current_driver_type)
                fos = np.clip((dac[:, 0] * fostd + fomean).astype("int"), 0, None)
                react_ratio = (fos - mean_fos - 1e-6) / (mean_fos + 1e-6)
                react_ratios.append(react_ratio)
                coupon_fos.append(fos)
                # probs.append(prob)
            react_ratios = np.clip(np.array(react_ratios), -10, 10)
            multi_coupon_react_ratios.append(react_ratios)
            # probs = np.array(probs)
            # react_ratio_mean = np.mean(react_ratios, axis=0)
            # norm_react_ratio = react_ratios / react_ratio_mean
            name = '{}/{}/mean-react_ratios-unit_cp:{}-day:{}'.format(self.name, 'ratio-level', unit_coupon, day)
            tester.simple_plot(name=name, data=react_ratios.T)
            react_ratios_std = np.std(react_ratios, axis=0)
            name = '{}/{}/mean-react_ratios-unit_cp:{}-day:{}'.format(self.name, 'hist-level', unit_coupon, day)
            tester.simple_hist( name, react_ratios_std.T, density=False, bins=50)

    def demer_data_gen(self, data_only):
        import os.path as osp
        dataset_path = osp.join(self.expert_path, self.folder_name,
                                'demer_rollout_%s_%d_%d_part_%d_unit_mask_%s_cost_%s_data_only_%s.csv' % (
                                self.city, self.start_date, self.end_date, self.partition_data, self.cp_unit_mask,
                                self.unit_cost_penlaty, data_only))
        if not osp.exists(dataset_path):
            standard_mean = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
            fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
            cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
            coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
            mean_fos = np.clip((standard_mean * fostd + fomean).astype("int"), 0, None)
            max_coupon_infos = []
            max_stand_cac = self.cac_max_value
            cp_unit_cost_number, cp_bound_number = self.gen_action_info()
            if data_only:
                cp_bound_number = 1
            dataset = np.empty(shape=(int(self.driver_number * (self.T - self.delay_buffer.max_delay) * cp_bound_number * cp_unit_cost_number),
                                      len(self.features_set) - self.ob_index_init_shift))
            counter = 0
            for day in range(self.delay_buffer.max_delay, self.T):
                not_delay_state = self.traj_real_obs[day].copy()
                for i in range(0, cp_bound_number):
                    for j in range(cp_unit_cost_number):
                        if self.cp_unit_mask:
                            unit_coupon = self.cac_mean_std[1, 0]
                        else:
                            unit_coupon = 1 + j * 1
                        if data_only:
                            i = self.traj_real_acs[day][:, 0]
                            stand_cp = (np.array([0, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                            coupon_action[:, 0] = i
                            coupon_action[:, 1:] = stand_cp[1:]
                        else:
                            stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                            coupon_action[:, ] = stand_cp
                        inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                        dac, value = self.pi.act(False, inner_state, coupon_action,
                                                 self.current_driver_type)
                        fos = np.clip(np.ceil(dac[:, 0] * fostd + fomean), 0, None)
                        if self.unit_cost_penlaty:
                            coupon_info = self.restore_coupon_info(coupon_action)
                            reach_driver = np.where(fos > coupon_info[:, 0])
                            fos[reach_driver] -= coupon_info[reach_driver][:, 0] * coupon_info[reach_driver][:, 1] / self.unit_gmv
                        data = np.concatenate([inner_state, coupon_action, np.expand_dims(fos, axis=1)], axis=1)
                        dataset[counter * self.driver_number: (counter + 1) * self.driver_number] = data
                        counter += 1
            df = pd.DataFrame(dataset)
            df.columns = self.features_set[self.ob_index_init_shift:]
            osp.exists(dataset_path)
        else:
            df = pd.read_csv(dataset_path, index_col=0)
        df = df.rename(columns={"cnt_order_y": "target"})
        feature_type_path = osp.join(self.expert_path, "all_feature_map.txt")
        # df.index.name = 'id'
        return df, feature_type_path

    def reshape_and_gen_action(self, flat_fo_pred):
        # driver -> cost per coupon-order -> coupon_bound -> city
        # a = np.array([1,1,1, 2,2,2, 10,10,10, 20, 20, 20, 100, 100, 100, 200, 200, 200, 1000, 1000, 1000, 2000, 2000, 2000])
        # a.reshape([4, 2, 3])
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        cp_unit_cost_number, cp_bound_number = self.gen_action_info()
        fo_pred = np.clip(flat_fo_pred.reshape([cp_bound_number * cp_unit_cost_number, self.driver_number]), 0, None)
        max_index = np.argmax(fo_pred, axis=0)
        max_bound_index = max_index // cp_unit_cost_number
        bound_value = np.array(range(0, cp_bound_number))
        max_bound_value = bound_value[max_bound_index]
        if self.cp_unit_mask:
            max_unit_cost_value = self.cac_mean_std[1, 0] * np.ones((self.driver_number))
        else:
            unit_cost_value = np.array(range(cp_unit_cost_number))
            max_unit_cost_index = max_index % cp_unit_cost_number
            max_unit_cost_value = unit_cost_value[max_unit_cost_index]
        coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
        coupon_action[:, 0] = max_bound_value
        coupon_action[:, 1] = max_unit_cost_value
        coupon_action[:, 2:] = np.array([0, 0, -999])
        stand_cp = (coupon_action - cpmean) / cpstd
        return stand_cp

    def gen_action_info(self):
        max_stand_cac = self.cac_max_value
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        max_cp_info = max_stand_cac * cpstd + cpmean
        # target dataset
        if self.cp_unit_mask:
            cp_unit_cost_number = 1
        else:
            cp_unit_cost_number = int((max_cp_info[1] - 1) / 1)
        cp_bound_number = int(max_cp_info[0])
        return int(cp_unit_cost_number), cp_bound_number

    def gen_all_sa(self, obs):
        standard_mean = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
        fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
        mean_fos = np.clip((standard_mean * fostd + fomean).astype("int"), 0, None)

        max_coupon_infos = []
        cp_unit_cost_number, cp_bound_number = self.gen_action_info()
        not_delay_state = obs
        dataset = np.empty(
            shape=(int(self.driver_number * cp_bound_number * cp_unit_cost_number),
                   len(self.features_set) - self.ob_index_init_shift - 1))
        counter = 0
        for i in range(0, cp_bound_number):
            for j in range(cp_unit_cost_number):
                if self.cp_unit_mask:
                    unit_coupon = self.cac_mean_std[1, 0]
                else:
                    # assert gen_action_info 是以0.5为间隔来计算的
                    unit_coupon = 1 + j * 1
                stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                coupon_action[:, ] = stand_cp
                inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                sa = np.concatenate([inner_state, coupon_action], axis=1)
                dataset[counter * self.driver_number: (counter + 1) * self.driver_number] = sa
                counter += 1

        df = pd.DataFrame(dataset)
        df.columns = self.features_set[self.ob_index_init_shift:-1]
        # df.index.name = 'id'
        return df

    def demer_optimal_test(self):
        standard_mean = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
        fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
        mean_fos = np.clip((standard_mean * fostd + fomean).astype("int"), 0, None)
        max_coupon_infos = []
        for day in range(self.delay_buffer.max_delay, self.T, 5):
            not_delay_state = self.traj_real_obs[day].copy()
            # delay_state = self.traj_real_obs[day - self.delay_buffer.max_delay]
            multi_coupon_react_ratios = []
            unit_coupon = self.cac_mean_std[1, 0]
            react_ratios = []
            coupon_fos = []
            # probs = []
            max_cp = 30
            for i in range(0, max_cp):
                stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                coupon_action[:, ] = stand_cp
                # res = self.pi.cact_prob(delay_state)
                # policy_mean, policy_std = res[0][0], res[0][1]
                # policy_logstd = np.log(policy_std)
                # neg_log_prob = 0.5 * np.sum(np.square((stand_cp - policy_mean) / policy_std), axis=-1) \
                #                + 0.5 * np.log(2.0 * np.pi) * stand_cp.shape[-1] \
                #                + np.sum(policy_logstd, axis=-1)
                # prob = np.exp(-1 * neg_log_prob)
                inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                dac, value = self.pi.act(False, inner_state, coupon_action,
                                         self.current_driver_type)
                fos = np.clip((dac[:, 0] * fostd + fomean).astype("int"), 0, None)
                coupon_fos.append(fos)

            coupon_fos = np.array(coupon_fos).T
            has_react_driver_idx = np.where(coupon_fos.std(axis=1) > 0)[0]
            coupon_fos = coupon_fos[has_react_driver_idx]
            filter = 'filter'

            max_coupon_info = np.argmax(coupon_fos, axis=1)
            name = '{}/{}-{}/cp-unit_cp:{}-day:{}'.format(self.name, 'hist-optimal-cp', filter, unit_coupon, day)
            tester.simple_hist(name, max_coupon_info, density=True, bins=max_cp)
            name = '{}/{}-{}/cp-unit_cp:{}-day:{}'.format(self.name, 'hist-all-optimal-cp', filter, unit_coupon, day)

            max_coupon_fos = np.max(coupon_fos, axis=1)
            not_react_driver = np.where(np.abs(coupon_fos - np.expand_dims(max_coupon_fos, axis=1)) < 1)[1]
            tester.simple_hist(name, not_react_driver, density=True, bins=max_cp)

            reverse_coupon_info = coupon_fos[:, ::-1]
            reverse_max_coupon_info = max_cp - np.argmax(reverse_coupon_info, axis=1)
            name = '{}/{}-{}/cp-unit_cp:{}-day:{}'.format(self.name, 'hist-max-optimal-cp', filter, unit_coupon, day)
            tester.simple_hist(name, reverse_max_coupon_info, density=True, bins=max_cp)

    def demer_prop_test(self):
        standard_mean = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
        fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
        cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        coupon_action = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
        mean_fos = np.clip((standard_mean * fostd + fomean).astype("int"), 0, None)
        for day in range(self.delay_buffer.max_delay, self.T, 5):
            not_delay_state = self.traj_real_obs[day].copy()
            # delay_state = self.traj_real_obs[day - self.delay_buffer.max_delay]
            multi_coupon_react_ratios = []
            for unit_coupon in np.arange(1, 5):
                react_ratios = []
                # probs = []
                for i in range(1, 50):
                    stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                    coupon_action[:, ] = stand_cp
                    # res = self.pi.cact_prob(delay_state)
                    # policy_mean, policy_std = res[0][0], res[0][1]
                    # policy_logstd = np.log(policy_std)
                    # neg_log_prob = 0.5 * np.sum(np.square((stand_cp - policy_mean) / policy_std), axis=-1) \
                    #                + 0.5 * np.log(2.0 * np.pi) * stand_cp.shape[-1] \
                    #                + np.sum(policy_logstd, axis=-1)
                    # prob = np.exp(-1 * neg_log_prob)
                    inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                    dac, value = self.pi.act(False, inner_state, coupon_action,
                                             self.current_driver_type)
                    fos = np.clip((dac[:, 0] * fostd + fomean).astype("int"), 0, None)
                    react_ratio = (fos - mean_fos - 1e-6) / (mean_fos + 1e-6)
                    react_ratios.append(react_ratio)
                    # probs.append(prob)
                react_ratios = np.clip(np.array(react_ratios), -10, 10)
                multi_coupon_react_ratios.append(react_ratios)
                # probs = np.array(probs)
                # react_ratio_mean = np.mean(react_ratios, axis=0)
                # norm_react_ratio = react_ratios / react_ratio_mean
                if unit_coupon % 1 == 0:
                    name = '{}/{}/react_ratios-unit_cp:{}-day:{}'.format(self.name, 'ratio-level', unit_coupon, day)
                    tester.simple_plot(name=name, data=react_ratios.T)
                    react_ratios_std = np.std(react_ratios, axis=0)
                    name = '{}/{}/react_ratios-std-unit_cp:{}-day:{}'.format(self.name, 'hist-level', unit_coupon, day)
                    tester.simple_hist( name, react_ratios_std.T, density=False, bins=50)
                    name = '{}/{}/react_ratios-unit_cp:{}-day:{}'.format(self.name, 'hist-level', unit_coupon, day)
                    tester.simple_hist( name, react_ratios.flatten(), density=False, bins=50)
            multi_coupon_react_ratios = np.array(multi_coupon_react_ratios).transpose([1, 0, 2])
            for cp_level in range(50):
                if cp_level % 5 == 0:
                    name = '{}/{}/react_ratios-unit_level:{}-day:{}'.format(self.name, 'ratio-coupon', cp_level, day)
                    tester.simple_plot(name=name, data=multi_coupon_react_ratios[cp_level].T)
                    react_ratios_std = np.std(multi_coupon_react_ratios[cp_level], axis=0)
                    name = '{}/{}/react_ratios-unit_cp:{}-day:{}'.format(self.name, 'hist-coupon', cp_level, day)
                    tester.simple_hist(name, react_ratios_std.T, density=False, bins=50)


    def make_expert_statistics(self):
        assert self.expert_info is not None
        spends = self.expert_info['daily_cost']
        total_fos = self.expert_info['daily_real_rews'] / self.unit_gmv
        coupon_ratio = spends / ((total_fos + 1e-6) * self.unit_gmv)
        coupon_ratio_filter = coupon_ratio[np.where(coupon_ratio > 0)]
        percentile = np.percentile(coupon_ratio_filter, [10, 33, 50, 66, 90, 95, 100])
        self.coupon_ratio = percentile[4] * self.coupon_ratio_scale
        logger.record_tabular("cp/bound", self.coupon_ratio)
        logger.record_tabular("cp/95-percent", percentile[5])
        logger.dump_tabular()
        logger.info("coupon info {}".format(self.coupon_ratio))

    def load_simulator(self, pi=None, reuse=False):
        import os.path as osp
        logger.info("ckpt path  {}".format(self.load_model_path))
        if pi is None:
            ckpt_path = self.load_model_path  # +'-%d'%(self.simulator_version)
            if self.sim_version != -1:
                ckpt_path = osp.join(ckpt_path, 'checkpoint-%d' % (self.sim_version))
            else:
                ckpt_path = tf.train.latest_checkpoint(ckpt_path)
            sess = tf.get_default_session()
            inner_low_state = np.zeros(self.dim_state - self.ob_index_init_shift)
            inner_high_state = np.ones(self.dim_state - self.ob_index_init_shift) * 50
            inner_observation_space = spaces.Box(low=inner_low_state, high=inner_high_state)
            self.pi = self.policy_fn(self.name + "pi", inner_observation_space, self.action_space, self.c_action_space,
                                     reuse=reuse)
            # sess.run(tf.global_variables_initializer())
            var_list =self.pi.get_variables()
            var_dict = {}
            for v in var_list:
                key_name = v.name[v.name.find(self.name) + len(self.name):]
                # key_name = '/'.join(v.name.split('/')[1:])
                key_name = key_name.split(':')[0]
                if 'qffc' in key_name and not tester.hyper_param['use_predict_q']:
                    continue
                var_dict[key_name] = v
            sess.run(tf.initialize_variables(var_list))
            saver = tf.train.Saver(var_list=var_dict)
            # U.initialize()
            logger.info("ckpt_path [load_simulator] {}".format(ckpt_path))
            # print("ckpt_path [load_simulator]: {}".format(ckpt_path))
            saver.restore(sess, ckpt_path)
        else:
            print("refer instead of create")
            self.pi = pi
        self.pi.coupon_action_info = self.cp_info
        max_iter = int(ckpt_path.split('-')[-1])
        assert max_iter > 0
        print("----------load {}-simulator model-{} successfully---------".format(self.name, max_iter))

    def zero_coupon_act(self, stochastic, ob, current_driver_type):
        return self.zero_coupon, None

    def real_act(self, obs, stochastic, current_driver_type):
        return self.traj_real_acs[self.timestep, :, :self.dim_coupon_feature], None

    def gen_driver_action(self, stochastic, not_delay_state, coupon_action, delay_state=None, use_expert_data=False,
                          optimal_acs_test=False):
        if use_expert_data:
            action = self.traj_real_acs[self.timestep, :, self.dim_coupon_feature:]
            return action, None
        if self.hand_code_driver_action:
            random_scale = self.get_random_sacle(obs_list=delay_state)
            # if self.timestep == 2 or self.timestep == self.T - 1:
            #     import os.path as osp
            #     save_path = osp.join(tester.results_dir, 'rand-{}-{}.png'.format(self.timestep, self.name))
            #     if not osp.exists(save_path):
            #         from matplotlib import pyplot as plt
            #         plt.cla()
            #         plt.hist(random_scale, bins=100, histtype='step',
            #                  label=['mean', 'std', 'max', 'operated', 'sensitive'])
            #         plt.legend(prop={'size': 8})
            #         logger.info("random_scale: ", random_scale)
            #         plt.savefig(save_path)
            if self.mdp_env:
                standard_fos = (delay_state[:, self.history_fos_index] - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
                max_stand_fos, mean_stand_fos, std_stand_fos, no_zero_fos_ratio = self._compute_statistics_info_merge(standard_fos)
            else:
                max_stand_fos = self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS]
                mean_stand_fos = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
                std_stand_fos = self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS]
            no_zero_fos_ratio = self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO]
            coupon_sensitive_ratio = self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO]
            def stat_test():
                if self.given_statistics:
                    stat_info = delay_state[:, self.statistics_index]
                    assert np.allclose(stat_info, np.stack([max_stand_fos, mean_stand_fos, std_stand_fos, no_zero_fos_ratio, coupon_sensitive_ratio], axis=1))
                if self.given_scale:
                    scale_info = delay_state[:, self.random_scale_index]
                    assert np.allclose(scale_info, random_scale)
            emb_unit_test_cont.add_test('test_constraint-{}-{}'.format(self.name, self.timestep),
                                        stat_test, emb_unit_test_cont.test_type.SINGLE_TIME)
            emb_unit_test_cont.do_test('test_constraint-{}-{}'.format(self.name, self.timestep))

            # np.clip(random_scale[:, 4] * self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO], 0, 1)
            driver_potential = max_stand_fos - mean_stand_fos

            # mean_stand_fos = random_scale[:, 0] * mean_stand_fos
            std_stand_fos = std_stand_fos
            # driver_potential = random_scale[:, 2] * driver_potential
            max_stand_fos = mean_stand_fos + driver_potential
            no_zero_fos_ratio = no_zero_fos_ratio  # np.clip(random_scale[:, 3] * no_zero_fos_ratio, 0, 1)
            coupon_sensitive_ratio = coupon_sensitive_ratio

            # TODO: 加一个单元测试，保证mean std 是当日的mean std这类的统计量

            coupon_info = self.restore_coupon_info(coupon_action, reshape_sec_fos=False, precision=3)
            max_fos = np.array(self.compute_fos(np.expand_dims(max_stand_fos, axis=1)))
            mean_fos = np.array(self.compute_fos(np.expand_dims(mean_stand_fos, axis=1)))
            std_fos = std_stand_fos * self.real_std_fos
            # max_fos += 1
            if self.hand_code_type == HandCodeType.MAX_TRANS:
                upper_bound_fos = max_fos + np.ceil(random_scale[:, ConstraintType.ORDER.index(ConstraintType.MAX_STANDARD_FOS)] + random_scale[:, ConstraintType.ORDER.index(ConstraintType.MEAN_STANDARD_FOS)])
                low_bound_fos = mean_fos + np.ceil(random_scale[:, ConstraintType.ORDER.index(ConstraintType.MEAN_STANDARD_FOS)])
            elif self.hand_code_type == HandCodeType.STD_TRANS:
                upper_bound_fos = np.ceil(mean_fos + std_fos + random_scale[:, ConstraintType.ORDER.index(ConstraintType.STD_STANDARD_FOS)])
                low_bound_fos = mean_fos
            else:
                raise NotImplementedError
            react_driver = np.where(np.logical_and(np.logical_and((upper_bound_fos - coupon_info[:, 0]) >= 0,
                                                   coupon_info[:, 1] - coupon_sensitive_ratio >= 0),
                                                   (coupon_info[:, 0] - low_bound_fos) >= 0))
            if optimal_acs_test:
                if react_driver[0].shape[0] != upper_bound_fos.shape[0]:
                    logger.info("react shape {}/{}".format(react_driver[0].shape[0], upper_bound_fos.shape[0]))
                    for ub, cp, lb in zip(upper_bound_fos, coupon_info[:, 0], low_bound_fos):
                        if ub < cp or cp < lb:
                            logger.info("ub {} > cp {} > lb {}".format(ub, cp, lb))
                    for cs, cp in zip(coupon_sensitive_ratio, coupon_info[:, 1]):
                        if cs > cp:
                            logger.info("cs {} > cp {}".format(cs, cp))
                    raise RuntimeError
            # react_driver = np.where(np.logical_and((max_fos - coupon_info[:, 0]) >= 0,
            #                                        coupon_info[:, 1] - coupon_sensitive_ratio >= 0))
            if (not stochastic) or self.deter_env:
                std_stand_fos = np.zeros(shape=std_stand_fos.shape)
                no_zero_fos_ratio = np.ones(shape=no_zero_fos_ratio.shape)
            standard_fos_sample = np.random.normal(mean_stand_fos, std_stand_fos)
            react_mean_fos = self.standardize_fos(coupon_info[:, 0])
            standard_fos_sample[react_driver] = np.random.normal(react_mean_fos[react_driver], std_stand_fos[react_driver] )
            zero_fos_index = np.where(np.random.random(mean_stand_fos.shape[0]) > no_zero_fos_ratio)
            standard_fos_sample[zero_fos_index] = self.standard_zero_fos
            dac = np.expand_dims(standard_fos_sample, axis=1)
            value = None
        else:
            inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
            dac, value = self.pi.act(stochastic or (not self.deter_env), inner_state, coupon_action, self.current_driver_type)
        invalid_reg_fos = np.where((dac[:, 0] - self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS]) > (self.standard_one_fos - self.standard_zero_fos))
        logger.record_tabular("env-{}/too_large_fos_ratio-{}".format(self.name, self.dynamic_type), invalid_reg_fos[0].shape[0] / self.driver_number)

        if self.constraint_driver_action:
            standard_mean, standard_std, no_zero_ratio = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS][invalid_reg_fos], self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS][invalid_reg_fos], self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO][invalid_reg_fos]
            zero_fos_index = np.where(np.random.random(invalid_reg_fos[0].shape[0]) > no_zero_ratio)
            revise_standard_fos = np.random.normal(standard_mean, standard_std)
            revise_standard_fos[zero_fos_index] = self.standard_zero_fos
            dac[invalid_reg_fos, 0] = revise_standard_fos
        return dac, value

    def get_optimal_act(self, obs_list, not_delay_state=None):
        if self.hand_code_driver_action:
            random_scale = self.get_random_sacle(obs_list=obs_list)
            if self.mdp_env:
                standard_fos = (obs_list[:, self.history_fos_index] - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
                max_stand_fos, mean_stand_fos, std_stand_fos, no_zero_fos_ratio = self._compute_statistics_info_merge(standard_fos)
            else:
                max_stand_fos = self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS]
                mean_stand_fos = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
                std_stand_fos = self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS]

            no_zero_fos_ratio = self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO]
            driver_potential = max_stand_fos - mean_stand_fos
            # mean_stand_fos = random_scale[:, 0] * mean_stand_fos
            std_stand_fos = std_stand_fos
            # driver_potential = random_scale[:, 2] * driver_potential

            max_stand_fos = mean_stand_fos + driver_potential
            # max_stand_fos = random_scale[:, 2] * max_stand_fos
            no_zero_fos_ratio = no_zero_fos_ratio # np.clip(random_scale[:, 3] * no_zero_fos_ratio, 0, 1)
            coupon_sensitive_ratio = self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO] # np.clip(random_scale[:, 4] * self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO], 0, 1)
            coupon_info = self.restore_coupon_info(self.zero_coupon, reshape_sec_fos=False, precision=3)
            max_fos = np.array(self.compute_fos(np.expand_dims(max_stand_fos, axis=1)))
            mean_fos = np.array(self.compute_fos(np.expand_dims(mean_stand_fos, axis=1)))
            std_fos = std_stand_fos * self.real_std_fos
            # max_fos += 1
            if self.hand_code_type == HandCodeType.MAX_TRANS:
                upper_bound_fos = max_fos + np.ceil(
                    random_scale[:, ConstraintType.ORDER.index(ConstraintType.MAX_STANDARD_FOS)] + random_scale[:,
                                                                                                   ConstraintType.ORDER.index(
                                                                                                       ConstraintType.MEAN_STANDARD_FOS)])
                low_bound_fos = mean_fos + np.ceil(
                    random_scale[:, ConstraintType.ORDER.index(ConstraintType.MEAN_STANDARD_FOS)])
            elif self.hand_code_type == HandCodeType.STD_TRANS:
                upper_bound_fos = np.ceil(mean_fos + std_fos + random_scale[:,
                                                       ConstraintType.ORDER.index(ConstraintType.STD_STANDARD_FOS)])
                low_bound_fos = mean_fos
            else:
                raise NotImplementedError

            coupon_info[:, 0] = upper_bound_fos
            coupon_info[:, 1] = coupon_sensitive_ratio + 0.1
            coupon_info = np.clip(coupon_info, None, self.coupon_info_bound)
            null_coupon_driver = np.where(coupon_info[:, 0] < low_bound_fos)
            if null_coupon_driver[0].shape[0] > 0:
                coupon_info[null_coupon_driver, 0] = 0
                coupon_info[null_coupon_driver, 1] = 0
        else:
            fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
            cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
            coupon_info = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
            max_cp = int(self.coupon_info_bound[0][0])
            # assert self.cp_unit_mask
            unit_coupon = self.cac_mean_std[1, 0]
            coupon_fos = []
            for i in range(0, max_cp):
                stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                coupon_info[:, ] = stand_cp
                inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                dac, value = self.pi.act(False, inner_state, coupon_info,
                                         self.current_driver_type)
                fos = np.clip((dac[:, 0] * fostd + fomean).astype("int"), 0, None)
                coupon_fos.append(fos)
            coupon_fos = np.array(coupon_fos).T
            # has_react_driver_idx = np.where(coupon_fos.std(axis=1) > 0)[0]
            max_coupon_info = np.argmax(coupon_fos, axis=1)
            coupon_info = coupon_info * cpstd + cpmean
            coupon_info[:, 0] = max_coupon_info
        rescale_action = self.cp_info.construct_coupon_info(coupon_info, sec_reshape=False, is_r_factor=True)

        return rescale_action

    def get_sub_optimal_act(self, obs_list, day_trace=-7, not_delay_state=None):
        if self.hand_code_driver_action:
            obs_number = obs_list.shape[0]
            random_scale = self.get_random_sacle(sub_opt=True, obs_list=obs_list)
            if self.mdp_env:
                standard_fos = (obs_list[:, self.history_fos_index] - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
                max_stand_fos, mean_stand_fos, std_stand_fos, no_zero_fos_ratio = self._compute_statistics_info_merge(standard_fos[:, day_trace:])
            else:
                max_stand_fos = self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS]
                mean_stand_fos = self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS]
                std_stand_fos = self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS]
            no_zero_fos_ratio = self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO]

            driver_potential = max_stand_fos - mean_stand_fos
            # mean_stand_fos = random_scale[:, 0] * mean_stand_fos
            std_stand_fos = std_stand_fos
            # driver_potential = random_scale[:, 2] * driver_potential
            max_stand_fos = mean_stand_fos + driver_potential
            no_zero_fos_ratio = no_zero_fos_ratio # np.clip(random_scale[:, 3] * no_zero_fos_ratio, 0, 1)
            coupon_sensitive_ratio = self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO]
            coupon_info = self.restore_coupon_info(self.zero_coupon, reshape_sec_fos=False, precision=3)
            max_fos = np.array(self.compute_fos(np.expand_dims(max_stand_fos, axis=1)))
            mean_fos = np.array(self.compute_fos(np.expand_dims(mean_stand_fos, axis=1)))
            std_fos = std_stand_fos * self.real_std_fos
            # max_fos += 1
            if self.hand_code_type == HandCodeType.MAX_TRANS:
                upper_bound_fos = max_fos + np.ceil(
                    random_scale[:, ConstraintType.ORDER.index(ConstraintType.MAX_STANDARD_FOS)] + random_scale[:,
                                                                                                   ConstraintType.ORDER.index(
                                                                                                       ConstraintType.MEAN_STANDARD_FOS)])
                low_bound_fos = mean_fos + np.ceil(
                    random_scale[:, ConstraintType.ORDER.index(ConstraintType.MEAN_STANDARD_FOS)])
            elif self.hand_code_type == HandCodeType.STD_TRANS:
                upper_bound_fos = mean_fos + std_fos + random_scale[:,
                                                       ConstraintType.ORDER.index(ConstraintType.STD_STANDARD_FOS)]
                low_bound_fos = mean_fos
            else:
                raise NotImplementedError
            coupon_info[:, 0] = upper_bound_fos
            coupon_info[:, 1] = coupon_sensitive_ratio + 0.1
            coupon_info = np.clip(coupon_info, None, self.coupon_info_bound)
            null_coupon_driver = np.where(coupon_info[:, 0] < mean_fos)
            coupon_info[null_coupon_driver, 0] = 0
            coupon_info[null_coupon_driver, 1] = 0
        else:
            fomean, fostd = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
            cpmean, cpstd = get_mean_std(True, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
            coupon_info = np.zeros(shape=self.traj_real_acs[0, :, :self.dim_coupon_feature].shape)
            max_cp = int(self.coupon_info_bound[0][0])
            # assert self.cp_unit_mask
            unit_coupon = self.cac_mean_std[1, 0]
            coupon_fos = []
            for i in range(0, max_cp):
                stand_cp = (np.array([i, unit_coupon, 0, 0, -999]) - cpmean) / cpstd
                coupon_info[:, ] = stand_cp
                inner_state = np.array(not_delay_state)[:, self.ob_index_init_shift:]
                dac, value = self.pi.act(False, inner_state, coupon_info,
                                         self.current_driver_type)
                fos = np.clip((dac[:, 0] * fostd + fomean).astype("int"), 0, None)
                coupon_fos.append(fos)
            coupon_fos = np.array(coupon_fos).T
            coupon_fos = coupon_fos[:, ::-1]
            # has_react_driver_idx = np.where(coupon_fos.std(axis=1) > 0)[0]
            max_coupon_info = np.argmax(coupon_fos, axis=1)
            max_coupon_info = max_cp - max_coupon_info
            coupon_info = coupon_info * cpstd + cpmean
            coupon_info[:, 0] = max_coupon_info

        rescale_action = self.cp_info.construct_coupon_info(coupon_info, sec_reshape=False, is_r_factor=True)

        return rescale_action


    def step(self, action=None, stochastic=True, cluster_ac=False, dynamic_type=DynamicType.POLICY,
             one_step=False, cost_constraint=False, sub_optimal_trace=0, policy_mean=None, policy_std=None, update_rew_scale=False):
        not_delay_state = self.delay_buffer.get_last_state(send_whether=False)
        self.dynamic_type = dynamic_type
        if dynamic_type == DynamicType.EXPERT or dynamic_type == DynamicType.EXPERT_DATA:
            action = self.traj_real_acs[self.timestep, :, :self.dim_coupon_feature]
        elif dynamic_type == DynamicType.POLICY:
            if self.re_construct_ac:
                restore_acs = self.restore_coupon_info(action, reshape_sec_fos=False)
                if tester.hyper_param['revise_step']:
                    restore_acs = self.pi.revise_cact(restore_acs, fo_reshape=False)
                action = self.cp_info.construct_coupon_info(restore_acs, sec_reshape=False, is_r_factor=True)
        elif dynamic_type == DynamicType.SUB_OPTIMAL:
            action = self.get_sub_optimal_act(np.array(self.state), day_trace=sub_optimal_trace, not_delay_state=not_delay_state)
        elif dynamic_type == DynamicType.NULL_COUPON:
            action = self.zero_coupon_act(None, None, None)[0]
        elif dynamic_type == DynamicType.OPTIMAL:
            action = self.get_optimal_act(self.state, not_delay_state=not_delay_state)
        else:
            raise NotImplementedError
        if self.budget_constraint:
            action[np.where(self.has_spended > (self.driver_budget))] = self.zero_coupon[0]

        # fos = max(0, int(round(dac[0] * self.real_std_fos + self.real_mean_fos, 0)))
        dac, _ = self.gen_driver_action(stochastic, np.array(not_delay_state), delay_state=self.state, coupon_action=action,
                                        use_expert_data=dynamic_type == DynamicType.EXPERT_DATA,
                                        optimal_acs_test=dynamic_type == DynamicType.OPTIMAL)
        fos = self.compute_fos(dac)
        # predict_gmv = self.compute_gmv(dac)
        avg_gmv = fos * self.unit_gmv
        fst_spend, sec_spend, coupon_spend, coupon_info = self.compute_spend(fos, action)
        if self.budget_constraint:
            self.has_spended += coupon_spend
        zero_dac, _ = self.gen_driver_action(False, np.array(not_delay_state), delay_state=self.state, coupon_action=self.zero_coupon,
                                             use_expert_data=dynamic_type == DynamicType.EXPERT_DATA)
        zero_fos = self.compute_fos(zero_dac)

        if cost_constraint:
            expert_cost = np.sum(self.expert_info['cost']) * self.budget_expand
            budget = expert_cost - self.already_cost
            daily_budget = budget / (self.T - self.timestep)
            if np.sum(coupon_spend) > daily_budget:
                inc_fos = fos - zero_fos
                sort_index = np.argsort(inc_fos)[::-1]
                select_cost = 0
                number = None
                for number, i in enumerate(sort_index):
                    next_cost = select_cost + coupon_spend[i]
                    if next_cost > daily_budget:
                        break
                    select_cost = next_cost
                action[sort_index[number:]] = self.zero_coupon[0, :]
                # reset action
                dac, _ = self.gen_driver_action(stochastic, np.array(not_delay_state), delay_state=self.state, coupon_action=action)
                fos = self.compute_fos(dac)
                # predict_gmv = self.compute_gmv(dac)
                avg_gmv = fos * self.unit_gmv
                fst_spend, sec_spend, coupon_spend, coupon_info = self.compute_spend(fos, action)
            self.already_cost += np.sum(coupon_spend)
        r_dac = self.traj_real_acs[self.timestep, :, self.dim_coupon_feature:]
        r_cac = self.traj_real_acs[self.timestep, :, :self.dim_coupon_feature]
        r_fos = self.compute_fos(r_dac)
        # r_gmv = self.compute_gmv(r_dac)
        rfst_spend, rsec_spend, r_coupon_spend, rcoupon_info = self.compute_spend(r_fos, r_cac)
        coupon_info = self.restore_coupon_info(action, reshape_sec_fos=False, precision=3)
        info = {
            "policy_spend": np.array([fst_spend, sec_spend, coupon_spend]),
            "real": {
                    Statistic.FOS: r_fos,
                    Statistic.AVG_GMV: r_fos * self.unit_gmv,
                    Statistic.SPEND: r_coupon_spend,
            },
            "coupon": coupon_info
        }
        coupon_ratio = self.coupon_ratio
        # compute reward
        info['gmv'] = fos * self.unit_gmv
        info['fos'] = fos
        info['null_coupon_rews'] = zero_fos * self.unit_gmv
        info['cost'] = coupon_spend
        info['revenue'] = fos * self.unit_gmv - coupon_spend
        info['coupon_reached'] = np.where(coupon_spend > 0)[0].shape[0] / (np.where(coupon_info[:, 0] > 0)[0].shape[0] + 1e-6)
        info['coupon_send'] = np.where(coupon_info[:, 0] > 0)[0].shape[0] / coupon_spend.shape[0]
        reward = fos * self.unit_gmv
        if not self.only_gmv:
            if self.unit_cost_penlaty:
                penalty = coupon_spend * (self.unit_gmv / (self.max_unit_cost + 0.01) * 0.9)
            else:
                penalty = coupon_spend / coupon_ratio
            reward = reward - penalty
        info['real_rews'] = reward
        self.rew_history.update(reward)
        terms = np.sum([self.UE_penalty, self.budget_constraint, True])
        terms_weight = 1 / terms
        if not self.gmv_rescale:
            reward = terms_weight * (reward) / (self.rew_history.std + 1e-6)

            # target_mean = np.array(np.expand_dims(self.state[:, self.UE_index[self.UE_features.index(ConstraintType.MEAN_STANDARD_FOS + '-ue')]], axis=1))
            # TODO
            # target_std = np.array(np.expand_dims(self.state[:, self.UE_index[self.UE_features.index(ConstraintType.STD_STANDARD_FOS + '-ue')]], axis=1))
        if self.UE_penalty:
            upper_bound = self.state[:, self.UE_index[self.UE_features.index('ub')]]
            lower_bound = self.state[:, self.UE_index[self.UE_features.index('lb')]]
            ue_penalty = np.max([np.clip(coupon_info[:, 0] - upper_bound, 0, None), np.clip(lower_bound - coupon_info[:, 0], 0, None)], axis=0)
            self.ue_penalty_history.update(ue_penalty)
            info['ue_penalty'] = ue_penalty
            if not self.gmv_rescale:
                ue_penalty = terms_weight * (ue_penalty) / (self.ue_penalty_history.std + 1e-6)
            else:
                ue_penalty = ue_penalty * self.unit_gmv
            reward -= ue_penalty
        else:
            info['ue_penalty'] = reward * 0
        if self.budget_constraint:
            cost_penalty = (self.has_spended > (self.driver_budget))
            self.budget_history.update(cost_penalty)
            info['budget_penalty'] = cost_penalty
            if not self.gmv_rescale:
                cost_penalty = terms_weight * (cost_penalty) / (self.budget_history.std + 1e-6)
                # self.has_spended > (self.driver_budget * self.budget_expand)
            else:
                cost_penalty = 0
            reward -= cost_penalty
        else:
            info['budget_penalty'] = reward * 0
        assert not self.kl_penalty
        info['kl'] = reward * 0
        # if self.kl_penalty and policy_std is not None:
        #     expert_res = self.pi.cact_prob(self.state)
        #     expert_mean, expert_std = expert_res[0][0], expert_res[0][1]
        #     expert_scale = self.pi.cac_max_value - self.pi.cac_min_value
        #     expert_bias = self.pi.cac_min_value
        #     expert_mean = (expert_mean - expert_bias) / expert_scale
        #     expert_std = expert_std / expert_scale
        #
        #     policy_logstd = np.log(policy_std)
        #     expert_logstd = np.log(expert_std)
        #     kl = np.sum(expert_logstd - policy_logstd + (np.square(policy_std) + np.square(policy_mean - expert_mean))
        #                 / (2.0 * np.square(expert_std) + 1e-6) - 0.5, axis=-1)
        #     info['kl'] = kl
        #     # logger.info("kl max {}".format(np.max(kl)))
        #     if update_rew_scale:
        #         self.kl_history.extend(kl)
        #     # logger.info("std rew: {}, kl {}".format(np.std(self.rew_history),  np.std(self.kl_history)))
        #     reward = 0.5 * reward / (np.std(self.rew_history) + 1e-6) - 0.5 * kl / (np.std(self.kl_history) + 1e-6)
        #
        # else:
        #     info['kl'] = reward * 0

        self.timestep += 1
        done = (self.timestep >= self.T)
        batch_done = np.repeat([done], reward.shape[0], axis=0)
        if done:
            # return np.array(self.state), true_reward, done, {"diff_driver": diff_driver, "diff_coupon": diff_coupon}
            self.need_reset = True
            return np.array(self.state), reward, batch_done, info
        if self.run_one_step:
            assert not self.hand_code_driver_action, "this change the shape."
            dac = self.traj_real_acs[self.timestep, :, self.dim_coupon_feature:] # effect of driver response does not write to state.
        if not self.delay_date:
            self.state = self.construct_state(dac, self.compute_fos, self.timestep, obs_list=self.state)
        else:
            state = self.construct_state(dac, self.compute_fos, self.timestep, obs_list=self.state)
            assert np.where(np.isnan(state))[0].shape[0] == 0
            self.state, cur_ts = self.delay_buffer.pop()
            assert cur_ts == self.timestep - self.delay_buffer.max_delay
            self.delay_buffer.append(state)
        if self.norm_obs:
            self.state = np.array(self._norm_obs(np.array(self.state)))
        else:
            self.state = np.array(self.state)

        if self.remove_hist_state:
            state = self.state.copy()
            state[:, self.history_fos_index] = 0
        else:
            state = self.state


        return state, reward / self.unit_gmv, batch_done, info

    def generate_actions_info(self):
        # new quality index
        logger.info("generate_actions_info")
        policy_quality = Statistic('real_data')
        ob = self.reset(percent=1)
        quality = policy_quality
        act_f = self.real_act
        for i in range(1):
            ts = 0
            logger.info("Now eval: {} epi-{}".format(quality.name, i))
            while True:
                logger.info("time step {} ".format(ts))
                cac, _ = act_f(False, ob, 0)
                ob, rew, new, info = self.step(cac, eval=True, stochastic=False, cluster_ac=False, eval_name=quality.name)
                quality.append(info)
                ts += 1
                if new:
                    # quality.new_episode()
                    break
        quality.reshape_episode()
        quality.compute_statistic('')
        return np.concatenate([np.expand_dims(quality.driver_fos_mean, axis=1), np.expand_dims(quality.driver_fos_std, 1)], axis=1)
