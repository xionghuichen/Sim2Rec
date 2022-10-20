'''
Environment for simulating driver state-action transition.
'''
import matplotlib

matplotlib.use('Agg')

import os.path as osp

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
import baselines.common.tf_util as U
from abc import ABCMeta, abstractmethod
from common.delay_buffer import DelayBuffer
from common.config import *
import time
from common.emb_unit_test import emb_unit_test_cont

class BaseDriverEnv(gym.Env):

    __metaclass__ = ABCMeta
    def __init__(self, name, folder_name, expert_path, city, start_date, end_date, trajectory_batch, driver_type_amount, reduce_param, update_his_fo,
                 constraint_driver_action, partition_data, norm_obs, delay_date, simp_state,
                 UE_penalty, budget_constraint,
                 given_scale=None, time_change_coef=None, scale_coef=None, driver_coef=None,
                 group_coef=None, hand_code_driver_action=None, hand_code_type=None,
                 given_statistics=None, cluster_type=None, samples_per_driver=None, budget_expand=1.5,
                 clus_param=None, eval_result_dir=''):
        # info param:
        self.name = name
        self.given_statistics = given_statistics
        self.budget_constraint = budget_constraint
        self.UE_penalty = UE_penalty
        self.folder_name = folder_name
        self.budget_expand = budget_expand
        self.simp_state = simp_state
        self.partition_data = partition_data
        self.given_scale = given_scale
        self.clus_param = clus_param
        self.cluster_type = cluster_type
        self.expert_path = expert_path
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        ms_path, cms_path, features_set_path, data_path, T = self.generate_path()
        self.data_path = data_path
        # env-type definition parameter
        self.constraint_driver_action = constraint_driver_action
        self.driver_type_amount = driver_type_amount
        self.update_his_fo = update_his_fo
        self.delay_date = delay_date
        # new param
        self.norm_obs = norm_obs
        self.reduce_param = reduce_param
        assert reduce_param
        self.trajectory_batch = trajectory_batch
        self.current_driver_type = 0
        # self.state_buffer = deque(maxlen=day_delay)
        # self.day_delay = day_delay
        self.result_path = eval_result_dir
        if not os.path.exists(self.result_path) and self.result_path is not '':
            os.makedirs(self.result_path)
        self.cms_path = cms_path
        self.dms_path = ms_path
        # real action 没有经过 resacle， 这里的rescale 只是针对 policy 的预测而言的
        self.cac_mean_std = np.genfromtxt(cms_path)
        self.dac_mean_std = np.genfromtxt(ms_path)
        # set zero value as min value.
        self.cac_min_value = (0 - self.cac_mean_std[:, 0]) / self.cac_mean_std[:, 1]
        if len(self.dac_mean_std.shape) == 1:
            self.dac_mean_std = np.expand_dims(self.dac_mean_std, axis=0)
        self.dac_min_value = (0 - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
        self.features_set_path = features_set_path
        self.features_set = pickle.load(open(features_set_path, 'rb'))
        self.constraint_types = ConstraintType.ORDER
        self.UE_features = ['ub', 'lb']
        self.UE_feature_number = len(self.UE_features)
        self.budget_features = ['budget_left']
        self.budget_feature_number = len(self.budget_features)
        self.contraint_features = [ct + '-scale' for ct in self.constraint_types] # + [ct +'-b' for ct in self.constraint_types]
        self.statistics_features = self.constraint_types # [ct + '-w' for ct in self.constraint_types] + [ct +'-b' for ct in self.constraint_types]
        # hard code env
        if self.given_scale:
            assert ConstraintType.NUMBER == 5
            self.features_set = self.contraint_features + self.features_set
            self.ob_index_init_shift = ConstraintType.NUMBER
        else:
            self.ob_index_init_shift = 0
        if self.given_statistics:
            self.features_set = self.statistics_features + self.features_set
            self.ob_index_init_shift += ConstraintType.NUMBER
        if self.UE_penalty:
            self.features_set = self.UE_features + self.features_set
            self.ob_index_init_shift += self.UE_feature_number
        if self.budget_constraint:
            self.features_set = self.budget_features + self.features_set
            self.ob_index_init_shift += self.budget_feature_number



        self.time_change_coef = time_change_coef
        self.scale_coef = scale_coef
        self.driver_coef = driver_coef
        self.group_coef = group_coef
        self.hand_code_driver_action = hand_code_driver_action
        self.hand_code_type = hand_code_type

        self.delay_buffer = DelayBuffer(self.features_set, has_driver_dim=True, more_delay=0)
        try:
            self.multi_driver = True
            self.driver_type_index = self.features_set.index('gulfstream_driver_features.driver_job_type')
        except Exception as e:
            self.multi_driver = False
            self.driver_type_index = -1

        self.dim_coupon_feature = 5
        self.dim_action = 1
        self.T = T
        if self.delay_date:
            self.iter_ts = self.T - self.delay_buffer.max_delay
        else:
            self.iter_ts = self.T
            self.delay_buffer.max_delay = 0
        self.dim_state = len(self.features_set) - self.dim_coupon_feature - self.dim_action
        self.dim_static_feature = self.dim_state - self.dim_action

        # observation
        self.low_state = np.zeros(self.dim_state)
        self.high_state = np.ones(self.dim_state) * 50

        # driver action
        self.low_action = np.zeros(self.dim_action)
        self.high_action = np.ones(self.dim_action) * 50

        # platform action
        self.low_c_action = np.zeros(self.dim_coupon_feature)
        self.high_c_action = np.ones(self.dim_coupon_feature) * 50

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        self.c_action_space = spaces.Box(low=self.low_c_action, high=self.high_c_action)
        # load real data
        self.real_mean_fos = self.dac_mean_std[0][0]
        self.real_std_fos = self.dac_mean_std[0][1]
        self.select_dids = None
        history_data = np.load(data_path)
        obs = history_data['obs']
        acs = history_data['acs']
        date_ids = history_data['date_ids']

        # format the state-action pair as trajectories of each driver.
        cmb_data = np.concatenate((date_ids, obs, acs), axis=1)
        self._cmb_df = pd.DataFrame(data=cmb_data).sort_values(by=[1, 0])  # columns 1: driver_id; column 2: date
        # confirm the date feature has been removed.

        # -------------track the index of some features, self.feature_set should be fix from here and on.
        # history fos index compute
        self.history_fos_index = []
        for i in range(29, 0, -1):
            current_day_cnt_col_name = 'gulfstream_driver_features.cnt_finish_orders_fast_{}_days'.format(i)
            self.history_fos_index.append(self.features_set.index(current_day_cnt_col_name))
        # weather index compute
        self.weather_set_index = []
        for i in range(len(self.features_set)):
            if 'date_type' in self.features_set[i] or 'caiyun.' in self.features_set[i]:
                self.weather_set_index.append(i)
        def feature_index_search(target_feature):
            index_found = []
            for i in range(len(self.features_set)):
                if self.features_set[i] in target_feature:
                    index_found.append(i)
            return index_found

        self.random_scale_index = feature_index_search(self.contraint_features)
        self.statistics_index = feature_index_search(self.statistics_features)
        self.to_zero_state_index = []
        if self.simp_state:
            for i in range(self.dim_static_feature):
                if i not in self.statistics_index and i not in self.random_scale_index and i not in self.history_fos_index:
                    self.to_zero_state_index.append(i)

                # -------------track the index of some features, self.feature_set should be fix from here and on.
        self.budget_constraint_index = feature_index_search(self.budget_features)
        self.UE_index = feature_index_search(self.UE_features)
        self.driver_feature_id = np.unique(np.array(self._cmb_df[1].values))
        self.traj_dataset = self._cmb_df.groupby([1])
        self.driver_ids = list(set(date_ids[:, 1]))
        self.driver_ids = self.driver_ids[:int(np.max([self.partition_data * len(self.driver_ids), 1]))]
        real_driver_type_amount = self._cmb_df[self.driver_type_index - self.ob_index_init_shift + 2].max() + 1
        if self.driver_type_amount < real_driver_type_amount:
            logger.warn(
                "[WARNING]: driver type amount ({}) is less than the real data ({})".format(self.driver_type_amount,
                                                                                            real_driver_type_amount))
        # sample percent
        if samples_per_driver is None:
            samples_per_driver = self.T - self.delay_buffer.max_delay
        else:
            samples_per_driver = min(self.T - self.delay_buffer.max_delay, samples_per_driver)
        driver_need = np.round(float(self.trajectory_batch) / samples_per_driver - 1)
        if driver_need > len(self.driver_ids):
            # driver_need = len(self.driver_ids)
            self.iter_ts = np.ceil(float(self.trajectory_batch) / driver_need / samples_per_driver) * samples_per_driver
        self.sample_percent = driver_need / len(self.driver_ids)
        self.num_envs = len(self.driver_ids)

        # coupon info
        from common.data_info import CouponActionInfo
        coupon_min_max = np.stack(
            [np.zeros(shape=self.cac_mean_std[:, 0].shape), self.cac_mean_std[:, 0] + 1.5 * self.cac_mean_std[:, 1]],
            axis=1)
        cp_info = CouponActionInfo(self.cac_mean_std, coupon_min_max)
        cp_info.configure(2)
        self.dac_max_value = np.array(self._cmb_df[list(range(self.features_set.index('cnt_order_y') - self.ob_index_init_shift + 2,
                               self.features_set.index('cnt_order_y') - self.ob_index_init_shift + self.dim_action + 2))].max())
        self.cac_max_value = np.array(self._cmb_df[list(range(self.features_set.index('coupon_feature_1')  - self.ob_index_init_shift + 2,
                               self.features_set.index('coupon_feature_1') - self.ob_index_init_shift + self.dim_coupon_feature + 2))].max())

        self.seed()
        self.unit_gmv = 20
        # outer init attributions

        self.group_matrix_w = None
        self.group_matrix_b = None
        self.driver_matrix_w = None
        self.driver_matrix_b = None
        if self.clus_param is not None:
            self.cluster_matrix_b =np.zeros(shape=(np.prod(self.clus_param), ConstraintType.NUMBER))
            self.cluster_matrix_w = np.zeros(shape=(np.prod(self.clus_param), ConstraintType.NUMBER))
        # dynamic attributions
        self.singleton_traj_dict = {}
        self.coupon_ratio = None
        self.driver_number = None
        self.driver_id = None
        self.bin_edges = None
        self.driver_cluster_hash = None
        self.cluster_features = None
        self.sub_cluster_index = None
        self.traj_real_obs, self.traj_real_acs = None, None
        self.traj_constraint_info = {}
        self.traj_constraint_scaled_w, self.traj_constraint_scaled_b = None, None
        self.zero_coupon = None
        self.timestep = None
        self.state = self.real_ob = None  # current state and real driver observation. TODO: change the meaning of state and ob.
        self.run_one_step = None
        self.city_coupon_sensitive_ratio = None
        self.already_cost = None
        self.obs_min = None
        self.obs_max = None
        self.unit_test_type = None
        self.need_reset = True
        # hand-code env dynamic attri
        self.feature_statistics = None
        self.replaced_feature_statistics = None
        self.time_diff_optimal_info = None
        self.city_diff_worst_info = None
        self.city_diff_best_info = None
        self.hand_code_clus_type_index = None
        self.hand_code_clus_type = None
        self.group_w = None
        self.replaced_env = self
        self.group_b = None


        # self.reset(percent=1)
        self.dynamic_info_gen()
        # print info
        self.cp_info = cp_info
        logger.info("[WARN] TODO: Now just set UNIT GMV to 20")
        logger.info("---- {} env info ---".format(self.name))
        logger.info("cac_min_value: ", self.cac_min_value, "dac_min_value: ", self.dac_min_value)
        logger.info("dim dim_static_feature: {}, dim_state {}".format( self.dim_static_feature, self.dim_state))
        logger.info('Driver numbers: ', len(self.driver_ids))
        logger.info("cac mean std :{}".format(self.cac_mean_std))
        logger.info("dac mean std :{}".format(self.dac_mean_std))
        logger.info("coupon ratio :{}".format(self.coupon_ratio))
        logger.info("coupon sensitive ratio :{}".format(self.city_coupon_sensitive_ratio))
        logger.info("---- {} env info ---".format(self.name))

    @property
    def standard_one_fos(self):
        return (1 - self.dac_mean_std[0, 0]) / self.dac_mean_std[0, 1]

    @property
    def standard_zero_fos(self):
        return self.dac_min_value[0]

    def generate_path(self):
        ms_path = osp.join(self.expert_path, self.folder_name,
                           "driver_response_mean_std_%s_%d_%d.txt" % (self.city, self.start_date, self.end_date))
        cms_path = osp.join(self.expert_path, self.folder_name,
                            "coupon_feature_mean_std_%s_%d_%d.txt" % (self.city, self.start_date, self.end_date))
        features_set_path = osp.join(self.expert_path, self.folder_name,
                                     'features_%s_%d_%d.pkl' % (self.city, self.start_date, self.end_date))
        expert_path = osp.join(self.expert_path, self.folder_name, "MDP_FadingDriver_SA_Dataset_%s_%d_%d.npz" % (
        self.city, self.start_date, self.end_date))
        T = compute_days(str(self.start_date), str(self.end_date), "%Y%m%d")
        return ms_path, cms_path, features_set_path, expert_path, T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def weight_gen(self, group_info, driver_info, time_change_coef=False):
        weight = (driver_info * self.driver_coef + group_info * self.group_coef) * self.scale_coef
        if time_change_coef:
            weight *= self.time_change_coef
        return weight

    def replace_env_scaled_info_gen(self, replaced_env):
        self.group_w = group_w = np.matmul(np.expand_dims(self.feature_statistics, axis=0), self.group_matrix_w)
        self.group_b = group_b = np.matmul(np.expand_dims(self.feature_statistics, axis=0), self.group_matrix_b)
        self.replaced_env = replaced_env
        rep_group_w = np.matmul(np.expand_dims(replaced_env.feature_statistics, axis=0), replaced_env.group_matrix_w)
        rep_group_b = np.matmul(np.expand_dims(replaced_env.feature_statistics, axis=0), replaced_env.group_matrix_b)
        number = 0
        for driver_id in self.select_dids:
            driver_info = self.singleton_traj_dict[driver_id]
            # driver_info[TrajInfo.SCALE_PARAM] = [w, b]
            # self.driver_matrix_w = np.random.normal(0, 1, size=(obs.shape[0], ConstraintType.NUMBER))
            rep_driver_w = self.driver_w[number]
            rep_driver_b = self.driver_b[number]
            number += 1
            rep_b = self.weight_gen(rep_group_b, rep_driver_b)
            rep_w = self.weight_gen(rep_group_w, rep_driver_w, time_change_coef=True)
            driver_info[TrajInfo.SCALE_PARAM][2:] = [rep_w, rep_b]

    def _scaled_info_gen(self):
        self.group_w = group_w = np.matmul(np.expand_dims(self.feature_statistics, axis=0), self.group_matrix_w)
        self.group_b = group_b = np.matmul(np.expand_dims(self.feature_statistics, axis=0), self.group_matrix_b)
        number = 0
        for driver_id in self.select_dids:
            driver_info = self.singleton_traj_dict[driver_id]
            # driver_info[TrajInfo.SCALE_PARAM] = [w, b]
            # self.driver_matrix_w = np.random.normal(0, 1, size=(obs.shape[0], ConstraintType.NUMBER))
            driver_w = self.driver_w[number]
            driver_b = self.driver_b[number]
            number += 1
            b = self.weight_gen(group_b, driver_b)
            w = self.weight_gen(group_w, driver_w, time_change_coef=True)
            driver_info[TrajInfo.SCALE_PARAM] = [w, b, w, b]

    def _norm_obs(self, obs):
        norm_obs = (obs - self.obs_min + 1e-6) / (self.obs_max - self.obs_min + 1e-6)
        return norm_obs

    def update_driver_weight(self, obs_list, select_env):
        if self.hand_code_driver_action:
            assert np.any(self.driver_coef) > 0
            if self.cluster_type == ClusterType.FOS:
                cluster_type_indices, cluster_type_hash = self.clustering_driver(obs_list)
                for key, cls_driver_index in cluster_type_hash.items():
                    key_digit = key.split('.')
                    sel_key = []
                    for hccti in self.hand_code_clus_type_index:
                        sel_key.append(int(key_digit[hccti]) - 1)
                    sel_key = tuple(sel_key)
                    self.driver_w[cls_driver_index] = select_env.cluster_matrix_w[sel_key]
                    self.driver_b[cls_driver_index] = select_env.cluster_matrix_b[sel_key]

    def get_random_sacle(self, sub_opt=False, timestep=None, obs_list=None):
        # return 1 + 0.5 * np.tanh(
        #     (self.traj_constraint_scaled_w * self.timestep + self.traj_constraint_scaled_b) / (self.T * 10))
        # if self.unit_test_type == UnitTestType.SUB_OPTIMAL_TEST:
        #     return np.ones(self.traj_constraint_scaled_w.shape)
        # else:
        replaced_env = self.replaced_env
        if timestep is None:
            timestep = self.timestep - self.delay_buffer.max_delay
        if self.unit_test_type == UnitTestType.SUB_OPTIMAL_TEST and sub_opt:
            if self.cluster_type == ClusterType.FOS:
                assert self.group_b is not None
                if self.driver_coef > 0:
                    self.update_driver_weight(obs_list=obs_list, select_env=replaced_env)
                    self.traj_constraint_scaled_b_sub = self.weight_gen(replaced_env.group_b, self.driver_b)
                    self.traj_constraint_scaled_w_sub = self.weight_gen(replaced_env.group_w, self.driver_w, time_change_coef=True)
            return 1 + 0.7 * np.abs((self.traj_constraint_scaled_w_sub * timestep) / (self.T) + self.traj_constraint_scaled_b_sub)
        else:
            if self.cluster_type == ClusterType.FOS:
                assert self.group_b is not None
                if self.driver_coef > 0:
                    self.update_driver_weight(obs_list=obs_list, select_env=self)
                    self.traj_constraint_scaled_b = self.weight_gen(self.group_b, self.driver_b)
                    self.traj_constraint_scaled_w = self.weight_gen(self.group_w, self.driver_w, time_change_coef=True)

            return 1 + 0.7 * np.abs((self.traj_constraint_scaled_w * timestep) / (self.T) + self.traj_constraint_scaled_b)

    def _base_reset(self):
        self.select_dids = select_dids = np.array(self.driver_ids)
        self.driver_id = select_dids
        self.driver_number = select_dids.shape[0]
        self.traj_real_obs = np.zeros(shape=(self.T, self.driver_number, self.dim_state))
        self.traj_real_acs = np.zeros(shape=(self.T, self.driver_number, self.dim_coupon_feature + self.dim_action))
        self.traj_constraint_info = {}
        self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
        self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
        self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
        self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO] = np.zeros(shape=(self.driver_number))
        self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO] = np.zeros(shape=(self.driver_number))
        self.traj_constraint_scaled_w = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
        self.traj_constraint_scaled_b = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
        self.traj_constraint_scaled_w_sub = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
        self.traj_constraint_scaled_b_sub = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
        self.zero_coupon = np.repeat([self.cac_min_value], self.driver_number, 0)
        for index, id in enumerate(select_dids):
            obs, acs, constraint, scale_params = self.get_traj_info_by_id(id)
            self.traj_real_obs[:, index, :] = obs
            self.traj_real_acs[:, index, :] = acs
            self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS][index] = constraint[
                ConstraintType.MAX_STANDARD_FOS]
            self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS][index] = constraint[
                ConstraintType.MEAN_STANDARD_FOS]
            self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS][index] = constraint[
                ConstraintType.STD_STANDARD_FOS]
            self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO][index] = constraint[
                ConstraintType.NO_ZERO_FOS_RATIO]
            self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO][index] = constraint[
                ConstraintType.COUPON_SENSITIVE_RATIO]
        if self.delay_date:
            self.delay_buffer.init_buf()
            for i in range(self.delay_buffer.max_delay):
                state = self.traj_real_obs[i].copy()
                if self.simp_state:
                    state[:, self.to_zero_state_index] = 0
                self.delay_buffer.append(state)

            self.timestep = self.delay_buffer.max_delay
            self.state, cur_ts = self.delay_buffer.pop()
            self.real_ob = self.state
            state = self.traj_real_obs[self.timestep].copy()
            if self.simp_state:
                state[:, self.to_zero_state_index] = 0
            self.delay_buffer.append(state)
        else:
            self.timestep = 0
            self.state = self.real_ob = self.traj_real_obs[0, :, :]

    def dynamic_info_gen(self):
        self._base_reset()
        # obs stats
        self.obs_min = self.traj_real_obs.min(axis=(0, 1))
        self.obs_max = self.traj_real_obs.max(axis=(0, 1))
        if self.given_scale: #TODO add given statis
            self.obs_min[self.random_scale_index] = 0
            self.obs_max[self.random_scale_index] = 1
        if self.given_statistics:
            self.obs_min[self.statistics_index] = 0
            self.obs_max[self.statistics_index] = 1
        # coupon_sensitive_ratio
        self.city_coupon_sensitive_ratio = np.mean(
                self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO][np.where(self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO] > 0)])
        spends = np.zeros(shape=(self.T - self.timestep, self.driver_number))
        total_fos = np.zeros(shape=(self.T - self.timestep, self.driver_number))
        for i in range(self.T - self.timestep):
            real_cac = self.traj_real_acs[i, :, :self.dim_coupon_feature]
            real_dac = self.traj_real_acs[i, :, self.dim_coupon_feature:]
            fos = self.compute_fos(real_dac)
            total_fos[i] = fos
            _, _, coupon_spend, coupon_info = self.compute_spend(fos, real_cac)
            spends[i] = coupon_spend
        coupon_ratio = spends / ((total_fos + 1e-6) * self.unit_gmv)
        coupon_ratio_filter = coupon_ratio[np.where(coupon_ratio > 0)]
        percentile = np.percentile(coupon_ratio_filter, [10, 33, 50, 66, 90, 99])
        self.coupon_ratio = percentile[3]
        # construct cluster bin.
        if self.clus_param is not None:
            cluster_features = []
            for stat in ClusterFeatures.ORDER:
                cluster_features.append(self.traj_constraint_info[stat])
            _, self.bin_edges = np.histogramdd(cluster_features, bins=tuple(self.clus_param))

    def hand_code_info_gen(self):
        self._base_reset()
        if self.cluster_type != ClusterType.FOS:
            assert self.driver_coef is not None
            if self.cluster_type == ClusterType.RAND:
                norm_obs = self._norm_obs(self.traj_real_obs[0])[:, self.to_zero_state_index]
                self.driver_w = np.matmul(norm_obs, self.driver_matrix_w)
                self.driver_b = np.matmul(norm_obs, self.driver_matrix_b)
            self._scaled_info_gen()

            mask_group_b = np.zeros(self.driver_b.shape)
            driver_weight = self.weight_gen(mask_group_b, self.driver_b)
            if self.hand_code_type == HandCodeType.MAX_TRANS:
                select_weight = driver_weight[:, [ConstraintType.ORDER.index(ConstraintType.MAX_STANDARD_FOS),
                                                  ConstraintType.ORDER.index(ConstraintType.MEAN_STANDARD_FOS)]].astype(np.int32)
            elif self.hand_code_type == HandCodeType.STD_TRANS:
                select_weight = driver_weight[:, [ConstraintType.ORDER.index(ConstraintType.STD_STANDARD_FOS)]].astype(np.int32)
            clus_param = (select_weight.max(axis=0) - select_weight.min(axis=0)).astype(np.int32)
            logger.record_tabular("env-{}/clusters".format(self.name), np.prod(clus_param))
            _, bin_edges = np.histogramdd(select_weight, bins=tuple(clus_param))
            sub_cluster_index = []
            for feature, edge in zip(select_weight.T, bin_edges):
                sub_cluster_index.append(np.digitize(feature, edge))
            self.sub_cluster_index = sub_cluster_index
            self.sub_cluster_dids = self.select_dids

    '''random sample one driver, then initialize enviroment state S_0 with his/her first state'''
    def reset(self, percent=None, driver_number=None, source_env_data=None, keep_did=False, run_one_step=False,
              unit_test_type=UnitTestType.NULL, scaled_budget=False):
        """

        PS: the statement with condition if xx is None is used for initialize environment attribution.

        :param percent:
        :param driver_number:
        :param source_env_data:
        :param keep_did:
        :param run_one_step:
        :return:
        """
        self.run_one_step = run_one_step
        self.unit_test_type = unit_test_type
        self.need_reset = False
        if percent is None:
            percent = self.sample_percent
        if driver_number is not None:
            shuffle_did = np.copy(self.driver_ids)
            np.random.shuffle(shuffle_did)
            select_dids = shuffle_did[:int(driver_number)]
        elif percent == 1:
            select_dids = np.array(self.driver_ids)
        else:
            shuffle_did = np.copy(self.driver_ids)
            np.random.shuffle(shuffle_did)
            select_dids = shuffle_did[:int(np.round(len(self.driver_ids) * (percent % 1)))]
            rounds = percent // 1
            for i in range((int(rounds))):
                select_dids = np.append(select_dids, shuffle_did)
        assert not keep_did
        if keep_did and self.select_dids is not None and select_dids.shape[0] == self.select_dids.shape[0]:
            logger.info("[reset] keep driver ids.")
            pass
        else:
            self.select_dids = select_dids
            self.driver_id = select_dids
            self.driver_number = select_dids.shape[0]
            self.traj_real_obs = np.zeros(shape=(self.T, self.driver_number, self.dim_state))
            self.traj_real_acs = np.zeros(shape=(self.T, self.driver_number, self.dim_coupon_feature + self.dim_action))
            self.traj_constraint_info = {}
            self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
            self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
            self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS] = np.zeros(shape=(self.driver_number))
            self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO] = np.zeros(shape=(self.driver_number))
            self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO] = np.zeros(shape=(self.driver_number))
            self.traj_constraint_scaled_w = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
            self.traj_constraint_scaled_b = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
            self.traj_constraint_scaled_w_sub = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
            self.traj_constraint_scaled_b_sub = np.zeros(shape=(self.driver_number, ConstraintType.NUMBER))
            self.zero_coupon = np.repeat([self.cac_min_value], self.driver_number, 0)
            for index, id in enumerate(select_dids):
                obs, acs, constraint, scale_params = self.get_traj_info_by_id(id)
                self.traj_real_obs[:, index, :] = obs
                self.traj_real_acs[:, index, :] = acs
                self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS][index] = constraint[ConstraintType.MAX_STANDARD_FOS]
                self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS][index] = constraint[ConstraintType.MEAN_STANDARD_FOS]
                self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS][index] = constraint[ConstraintType.STD_STANDARD_FOS]
                self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO][index] = constraint[ConstraintType.NO_ZERO_FOS_RATIO]
                self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO][index] = constraint[ConstraintType.COUPON_SENSITIVE_RATIO]
                if scale_params != []:
                    self.traj_constraint_scaled_w[index] = scale_params[0]
                    self.traj_constraint_scaled_b[index] = scale_params[1]
                    self.traj_constraint_scaled_w_sub[index] = scale_params[2]
                    self.traj_constraint_scaled_b_sub[index] = scale_params[3]
            self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO][
                np.where(self.traj_constraint_info[
                             ConstraintType.COUPON_SENSITIVE_RATIO] < 0)] = self.city_coupon_sensitive_ratio

        # embedding unit test ---
        def test_constraint_info():
            his_fos = self.traj_real_obs[self.delay_buffer.max_delay - 1, :, ][:, self.history_fos_index]
            mean, std = get_mean_std(True, mean=self.real_mean_fos, std=self.real_std_fos)
            standard_fos = (his_fos - mean) / std  # self.standardize_fos(np.expand_dims(his_fos, axis=1))
            max_standard_fos, mean_standard_fos, std_standard_fos, no_zero_fos_ratio = self._compute_statistics_info_merge(
                standard_fos)
            assert np.allclose(self.traj_constraint_info[ConstraintType.MAX_STANDARD_FOS], max_standard_fos) and \
                np.allclose(self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS], mean_standard_fos) and \
                np.allclose(self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS], std_standard_fos) and \
                np.allclose(self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO], no_zero_fos_ratio), \
                "error city name:{}".format(self.name)
        emb_unit_test_cont.add_test('test_constraint-{}'.format(self.name),
                                    test_constraint_info, emb_unit_test_cont.test_type.SINGLE_TIME)
        emb_unit_test_cont.do_test('test_constraint-{}'.format(self.name))
        # embedding unit test ---
        logger.info("sample city {} driver number: {}".format(self.name, self.driver_number))
        # init traj_real_obs
        # random scale + statistic
        if self.given_scale:
            for t in range(self.delay_buffer.max_delay + 1):
                random_scale = self.get_random_sacle(timestep=t, obs_list=self.traj_real_obs[t])
                self.traj_real_obs[t][:, self.random_scale_index] = random_scale
        if self.given_statistics:
            for t in range(self.delay_buffer.max_delay + 1):
                mean_standard_fos, std_standard_fos, max_standard_fos, no_zero_fos_ratio, coupon_sensitive_ratio = self.get_feature_statistics(self.traj_real_obs[t])
                self.traj_real_obs[t][:, self.statistics_index] = np.stack([max_standard_fos, mean_standard_fos, std_standard_fos, no_zero_fos_ratio, coupon_sensitive_ratio], axis=1)
        # ue feature
        if self.UE_penalty:
            upper_bound = self.compute_fos(np.expand_dims(self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS] + 1 * self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS], axis=1))
            lower_bound = self.compute_fos(np.expand_dims(self.traj_constraint_info[ConstraintType.MEAN_STANDARD_FOS] - 1 * self.traj_constraint_info[ConstraintType.STD_STANDARD_FOS], axis=1))
            for UE_idx, UE_f in zip(self.UE_index, self.UE_features):
                if UE_f == 'ub':
                    self.traj_real_obs[:, :, UE_idx] = upper_bound
                elif UE_f == 'lb':
                    self.traj_real_obs[:, :, UE_idx] = lower_bound
                else:
                    raise NotImplementedError
        if self.delay_date:
            self.timestep = self.delay_buffer.max_delay
        else:
            self.timestep = 0
        # budget feature
        spends = np.zeros(shape=(self.T - self.timestep, self.driver_number))
        total_fos = np.zeros(shape=(self.T - self.timestep, self.driver_number))
        for i in range(self.T - self.timestep):
            real_cac = self.traj_real_acs[i, :, :self.dim_coupon_feature]
            real_dac = self.traj_real_acs[i, :, self.dim_coupon_feature:]
            fos = self.compute_fos(real_dac)
            total_fos[i] = fos
            _, _, coupon_spend, coupon_info = self.compute_spend(fos, real_cac)
            spends[i] = coupon_spend
        self.driver_budget = np.sum(spends, axis=0) * self.budget_expand
        if scaled_budget:
            self.driver_budget = np.random.random(size=self.driver_budget.shape) * 3 * self.driver_budget
        if self.budget_constraint:
            assert len(self.budget_features) == 1
            self.traj_real_obs[:, :, self.budget_constraint_index] = np.expand_dims(self.driver_budget, axis=1)

        self.already_cost = 0
        if self.delay_date:
            self.delay_buffer.init_buf()
            for i in range(self.delay_buffer.max_delay):
                state = self.traj_real_obs[i].copy()
                if self.simp_state:
                    state[:, self.to_zero_state_index] = 0
                self.delay_buffer.append(state)


            self.state, cur_ts = self.delay_buffer.pop()
            self.real_ob = self.state
            state = self.traj_real_obs[self.timestep].copy()
            if self.simp_state:
                state[:, self.to_zero_state_index] = 0
            self.delay_buffer.append(state)
        else:
            self.timestep = 0
            self.state = self.real_ob = self.traj_real_obs[0, :, :].copy()
        self.update_driver_weight(self.state, select_env=self)

        self.has_spended = np.zeros(shape=self.driver_budget.shape)
        # clustering
        # multi-type driver setting unit test.
        if self.multi_driver:
            self.current_driver_type = self.real_ob[:, self.driver_type_index].astype('int32')
            assert (np.all(0 <= self.current_driver_type))
            if np.any(self.driver_type_amount <= self.current_driver_type):
                if random.randint(0, 40) == 1:
                    logger.warn(
                        "[WARNING]: driver type amount ({}) is less than the real data ({})".format(
                            self.driver_type_amount,
                            np.max(self.current_driver_type) + 1))
                self.current_driver_type[np.where(self.driver_type_amount <= self.current_driver_type)] = self.driver_type_amount - 1

        if self.norm_obs:
            return np.array(self._norm_obs(self.state))
        else:
            return np.array(self.state)

    def clustering_hash(self, obs_list, hidden_state=None):
        assert len(obs_list.shape) == 2
        obs_number = obs_list.shape[0]
        steps = obs_list.shape[0] / self.driver_number
        assert steps % 1 == 0
        if self.cluster_type == ClusterType.FOS:
            standard_fos = (obs_list[:, self.history_fos_index] - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
            traj_constraint_info = self._compute_statistics_info_merge(standard_fos, dict_ret=True)
            cluster_features = []
            for stat in ClusterFeatures.ORDER:
                cluster_features.append(traj_constraint_info[stat])
            sub_cluster_index = []
            for feature, edge in zip(cluster_features, self.bin_edges):
                sub_cluster_index.append(np.digitize(feature, edge))
        elif self.cluster_type == ClusterType.MAP or self.cluster_type == ClusterType.RAND:
            sub_cluster_index = self.sub_cluster_index

        def dim_test():
            for i in range(self.driver_number):
                assert obs_list[int(i * steps)][self.features_set.index('gulfstream_driver_features.gender')] == \
                       obs_list[int(i * steps) + 1][self.features_set.index('gulfstream_driver_features.gender')]
        emb_unit_test_cont.add_test('dim_test',
                                    dim_test, emb_unit_test_cont.test_type.SINGLE_TIME)
        emb_unit_test_cont.do_test('dim_test')
        driver_cluster_hash = {}
        def _hash_encode(didx):
            code = []
            for sub_cluster in sub_cluster_index:
                if self.cluster_type == ClusterType.FOS:
                    code.append(str(sub_cluster[didx]))
                else:
                    # TODO: assert didx is a step-first array.
                    code.append(str(sub_cluster[list(self.sub_cluster_dids).index(self.select_dids[int(didx / steps)])]))
            return '.'.join(code)
        for i in range(obs_number):
            if _hash_encode(i) not in driver_cluster_hash:
                driver_cluster_hash[_hash_encode(i)] = []
            driver_cluster_hash[_hash_encode(i)].append(i)
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
        logger.record_tabular("env-{}/cluster_small_than_3".format(self.name), small_cluster_count / cluster_number)
        logger.record_tabular("env-{}/clusters".format(self.name), len(driver_cluster_hash.keys()))
        return driver_cluster_list

    def clustering_driver(self, obs_list):
        assert len(obs_list.shape) == 2
        obs_number = obs_list.shape[0]
        driver_cluster_hash = self.clustering_hash(obs_list)
        driver_cluster_list = self.hash_to_array(obs_number, driver_cluster_hash)
        return driver_cluster_list, driver_cluster_hash

    def get_feature_statistics(self, obs):
        standard_fos = (obs[:, self.history_fos_index] - self.dac_mean_std[:, 0]) / self.dac_mean_std[:, 1]
        max_standard_fos, mean_standard_fos, std_standard_fos, no_zero_fos_ratio = self._compute_statistics_info_merge(
            standard_fos)
        coupon_sensitive_ratio = self.traj_constraint_info[ConstraintType.COUPON_SENSITIVE_RATIO]
        no_zero_fos_ratio = self.traj_constraint_info[ConstraintType.NO_ZERO_FOS_RATIO]
        return mean_standard_fos, std_standard_fos, max_standard_fos, no_zero_fos_ratio, coupon_sensitive_ratio

    def _compute_statistics_info_merge(self, standard_fos, dict_ret=False):
        standard_fos = standard_fos.copy()
        standard_fos = np.append(standard_fos,
                                 np.expand_dims(np.repeat(self.standard_one_fos, standard_fos.shape[0], axis=0),
                                                axis=1), axis=1)
        not_zero_fos_index = np.where(standard_fos >= self.standard_one_fos)
        zero_fos_index = np.where(standard_fos < self.standard_one_fos)
        standard_fos[:, -1] = self.standard_zero_fos # 最后一个呗临时置为1 的完单重置为0，但是其index被判定为有完单，避免有完单部分为空
        # compute ratio
        standard_fos_temp = standard_fos.copy()
        standard_fos_temp[zero_fos_index] = 0
        standard_fos_temp[not_zero_fos_index] = 1
        not_zero_fos_ratio = np.mean(standard_fos_temp, axis=1)
        # compute mean
        standard_fos[zero_fos_index] = 0
        mean_standard_fos = np.mean(standard_fos, axis=1)
        mean_standard_fos = mean_standard_fos / not_zero_fos_ratio  # 基数转化
        mean_standard_fos[np.where(mean_standard_fos == 0)] = self.standard_zero_fos
        # compute std
        standard_fos[zero_fos_index] = mean_standard_fos[zero_fos_index[0]]  # 所有0完单的位置，替换成其司机的完单均值
        std_standard_fos = np.std(standard_fos, axis=1)
        std_standard_fos = std_standard_fos / np.sqrt(not_zero_fos_ratio)  # 基数转化
        # compute max
        standard_fos[zero_fos_index] = self.standard_zero_fos # 所有0完单的位置，恢复期真实值
        max_standard_fos = np.max(standard_fos, axis=1)
        if dict_ret:
            info_dict = {}
            info_dict[ConstraintType.MAX_STANDARD_FOS] = max_standard_fos
            info_dict[ConstraintType.STD_STANDARD_FOS] = std_standard_fos
            info_dict[ConstraintType.MEAN_STANDARD_FOS] = mean_standard_fos
            info_dict[ConstraintType.NO_ZERO_FOS_RATIO] = not_zero_fos_ratio
            return info_dict
        else:
            return max_standard_fos, mean_standard_fos, std_standard_fos, not_zero_fos_ratio

    def get_traj_info_by_id(self, selected_dids):
        if selected_dids not in self.singleton_traj_dict:
            traj_data = self.traj_dataset.get_group(selected_dids).values[:, 2:]
            if self.ob_index_init_shift > 0:
                traj_data = np.concatenate([np.zeros(shape=(traj_data.shape[0], self.ob_index_init_shift)), traj_data], axis=1)

            obs = traj_data[:, :self.dim_state]
            acs = traj_data[:, self.dim_state:]
            self.singleton_traj_dict[selected_dids] = {}
            self.singleton_traj_dict[selected_dids][TrajInfo.OBS] = obs
            self.singleton_traj_dict[selected_dids][TrajInfo.ACS] = acs
            self.singleton_traj_dict[selected_dids][TrajInfo.CONSTRAINT] = {}
            self.singleton_traj_dict[selected_dids][TrajInfo.SCALE_PARAM] = []

            assert self.dim_action == 1
            # TOOD: coupon_avg_fos should be computed with historical data.
            coupon_info = self.restore_coupon_info(acs[:, :self.dim_coupon_feature], reshape_sec_fos=False)
            fos = self.compute_fos(acs[:, self.dim_coupon_feature:])
            coupon_avg_fos = coupon_info[np.where(np.logical_and(fos > coupon_info[:, 0], coupon_info[:, 0] > 0))[0], 1]
            # historical fos
            historical_fos = obs[self.delay_buffer.max_delay - 1, self.history_fos_index]
            standard_fos = self.standardize_fos(historical_fos)
            standard_fos = np.append(standard_fos, [self.standard_one_fos], axis=0)

            not_zero_fos_index = np.where(standard_fos >= self.standard_one_fos)
            standard_not_zero_fos = standard_fos[not_zero_fos_index]
            # avoid the error of empty not-zero fos
            standard_not_zero_fos[-1] = self.standard_zero_fos
            constraint_info = {
                ConstraintType.MEAN_STANDARD_FOS: np.mean(standard_not_zero_fos),
                ConstraintType.STD_STANDARD_FOS: np.std(standard_not_zero_fos),
                ConstraintType.MAX_STANDARD_FOS: np.max(standard_not_zero_fos),
                ConstraintType.NO_ZERO_FOS_RATIO: not_zero_fos_index[0].shape[0] / standard_fos.shape[0],
                ConstraintType.COUPON_SENSITIVE_RATIO: np.mean(coupon_avg_fos) if coupon_avg_fos.shape[0] > 0  else -1
            }
            self.singleton_traj_dict[selected_dids][TrajInfo.CONSTRAINT] = constraint_info
        obs = self.singleton_traj_dict[selected_dids][TrajInfo.OBS]
        acs = self.singleton_traj_dict[selected_dids][TrajInfo.ACS]
        constraint = self.singleton_traj_dict[selected_dids][TrajInfo.CONSTRAINT]
        scaled_param = self.singleton_traj_dict[selected_dids][TrajInfo.SCALE_PARAM]
        return obs, acs, constraint, scaled_param


    @abstractmethod
    def step(self, action, *args, **kwargs):
        pass

    def restore_coupon_info(self, action, rescale=True, reshape_sec_fos=True, precision=2):
        mean, std = get_mean_std(rescale, mean=self.cac_mean_std[:, 0], std=self.cac_mean_std[:, 1])
        coupon_info = action * np.expand_dims(std, 0) + np.expand_dims(mean, 0)
        # coupon_info = [action[:, i] * std[i] + mean[i] for i in range(self.dim_coupon_feature)]
        if reshape_sec_fos:
            coupon_info[:, 2] = np.round(coupon_info[:, 0] + coupon_info[:, 2], 0)
            coupon_info[:, 3] = np.round(coupon_info[:, 1] + coupon_info[:, 3], precision)
        else:
            coupon_info[:, 2] = np.round(coupon_info[:, 2], 0)
            coupon_info[:, 3] = np.round(coupon_info[:, 3], precision)
        coupon_info[:, 0] = np.round(coupon_info[:, 0], 0)
        coupon_info[:, 1] = np.round(coupon_info[:, 1], precision)
        coupon_info[:, 4] = np.clip(coupon_info[:, 4], None, 1)
        # 将不合理补贴清零：
        first_layer_reshape = np.where(np.logical_or(coupon_info[:, 0] <= 0, coupon_info[:, 1] <= 0))
        coupon_info[first_layer_reshape, :] = 0
        second_layer_reshape = np.where(np.logical_or(coupon_info[:, 2] <= 0, coupon_info[:, 3] <= 0))
        coupon_info[second_layer_reshape, 2:4] = 0
        # assert not np.any(np.where(coupon_info[:, 1] > 0 & coupon_info[:, 0] == 0))

        # first_layer_zero_idx = np.where(coupon_info[:, 0] <= 0)
        # 第一层如果是0， 用第二层进行替换
        # if reshape_sec_fos:
        #     coupon_info[first_layer_zero_idx, 0] = coupon_info[first_layer_zero_idx, 2]
        #     coupon_info[first_layer_zero_idx, 1] = coupon_info[first_layer_zero_idx, 3]
        # else:
        #     coupon_info[first_layer_zero_idx, 0] = coupon_info[first_layer_zero_idx, 2]
        #     coupon_info[first_layer_zero_idx, 1] = coupon_info[first_layer_zero_idx, 3]
        #     coupon_info[first_layer_zero_idx, 2] = 0
        #     coupon_info[first_layer_zero_idx, 3] = 0

        coupon_info = np.clip(coupon_info, 0, None)
        if reshape_sec_fos:
            rule_len = coupon_info[:, 2] - coupon_info[:, 0]
        else:
            rule_len = coupon_info[:, 2]

        r = coupon_info[:, 4]
        levels = np.clip(np.round(1 / (1 - r + 0.001), 0).astype('int32'),  1, rule_len) + 1
        coupon_info[:, 4] = 1 - 1 / (levels - 1 + 0.001)
        return coupon_info

    def standardize_fos(self, acs, rescale=True):
        mean, std = get_mean_std(rescale, mean=self.real_mean_fos, std=self.real_std_fos)
        return (acs - mean) / std

    def compute_fos(self, acs, rescale=True):
        mean, std = get_mean_std(rescale, mean=self.real_mean_fos, std=self.real_std_fos)
        return np.clip(np.round(acs[:, 0] * std + mean, 0).astype('int32'), 0, None)

    # def compute_gmv(self, acs, rescale=True):
    #     mean, std = get_mean_std(rescale, mean=self.dac_mean_std[4][0], std=self.dac_mean_std[4][1])
    #     return np.clip(np.round(acs[:, 4] * std + mean, 2).astype('int32'), 0, None)

    def construct_state(self, dac, compute_fos_func, timestep, driver_ids=None, obs_list=None):
        if self.run_one_step:
            return self.traj_real_obs[timestep].copy()
        if driver_ids is None:
            state = self.state.copy()  # np.zeros(shape=(self.select_dids.shape[0], self.dim_state))
            real_ob = self.traj_real_obs[timestep].copy()
        else:
            state = self.state[driver_ids].copy()  # np.zeros(shape=(driver_ids.shape[0], self.dim_state))
            real_ob = self.traj_real_obs[timestep, driver_ids].copy()
        # update history fos
        if self.update_his_fo:
            current_hist_index = self.history_fos_index[0]
            for his_fos in self.history_fos_index[1:]:
                real_ob[:, current_hist_index] = state[:, his_fos]
        state[:, self.dim_static_feature:] = dac
        if self.update_his_fo:
            real_ob[:, self.history_fos_index[-1]] = compute_fos_func(state[:, self.dim_static_feature:])
        state[:, :self.dim_static_feature] = real_ob[:, :self.dim_static_feature]
        if self.given_statistics:
            mean_standard_fos, std_standard_fos, max_standard_fos, no_zero_fos_ratio, coupon_sensitive_ratio = self.get_feature_statistics(state)
            state[:, self.statistics_index] = np.stack([max_standard_fos, mean_standard_fos, std_standard_fos, no_zero_fos_ratio, coupon_sensitive_ratio], axis=1)
        if self.given_scale:
            random_scale = self.get_random_sacle(timestep=timestep, obs_list=state)
            state[:, self.random_scale_index] = random_scale
        if self.budget_constraint:
            state[:, self.budget_constraint_index] = np.expand_dims(self.driver_budget - self.has_spended, axis=1)
        if self.simp_state:
            state[:, self.to_zero_state_index] = 0
        return state

    def compute_spend(self, fos, coupon_action, rescale=True,):
        # assert len(coupon_action) == self.dim_coupon_feature
        coupon_info = self.restore_coupon_info(coupon_action, rescale)
        first_fos = coupon_info[:, 0].astype('int32')
        first_bns = coupon_info[:, 1]
        first_spend = first_fos * first_bns
        sec_fos = coupon_info[:, 2].astype('int32')
        sec_bns = coupon_info[:, 3]
        rule_len = sec_fos - first_fos
        r = coupon_info[:, 4]
        levels = np.clip(np.round(1 / (1 - r + 0.001), 0).astype('int32'), 1, rule_len) + 1
        # levels = max(2, min(rule_len + 1, int(round(1 / (1 - r + 0.001)))))
        stepsize = np.clip(np.round(rule_len / (levels - 0.9999), 0), 0, None).astype('int32')
        sec_fos = np.max([sec_fos, first_fos], axis=0)
        fst_spends = np.zeros(shape=(fos.shape[0]))
        sec_spends = np.zeros(shape=(fos.shape[0]))
        coupon_spends = np.zeros(shape=(fos.shape[0]))
        for index, fo, f_fo, f_spend, sec_fo, sec_bn, step, rule in zip(list(range(self.select_dids.shape[0])),
                                                                        fos, first_fos, first_spend, sec_fos, sec_bns, stepsize, rule_len):
            fst_spend, sec_spend, coupon_spend = self.single_compute_spend(fo, f_fo, f_spend, sec_fo, sec_bn, step, rule)
            fst_spends[index] = fst_spend
            sec_spends[index] = sec_spend
            coupon_spends[index] = coupon_spend
        return fst_spends, sec_spends, coupon_spends, coupon_info

    def single_compute_spend(self, fos, first_fos, first_spend, sec_fos, sec_bns, stepsize, rule_len, ):
        spend_money_list = [0] * (sec_fos + 1)
        for fo in range(first_fos, sec_fos, max(1, stepsize)):
            for i in range(stepsize):
                # print(sec_fos, min(fo+i, sec_fos))
                spend_money_list[min(fo + i, sec_fos)] = first_spend + (fo - first_fos) * sec_bns
        spend_money_list[sec_fos] = first_spend + rule_len * sec_bns
        fst_spend = 1 if fos >= first_fos and first_fos > 0 else 0
        sec_spend = 1 if fos >= sec_fos and sec_fos > 0 else 0
        coupon_spend = spend_money_list[fos] if fos <= sec_fos else spend_money_list[-1]
        return fst_spend, sec_spend, coupon_spend
