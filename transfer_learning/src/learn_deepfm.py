
import pandas as pd
from transfer_learning.src.driver_simenv_disc import DriverEnv, MultiCityDriverEnv
from transfer_learning.src.DeepFM import DeepFM
import numpy as np
from common import logger
from common.time_step import time_step_holder
from common.config import *

class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["target"].values.tolist()
            # dfi.drop(["id", "target"], axis=1, inplace=True)
        # else:
        #     ids = dfi["id"].values.tolist()
        #     dfi.drop(["id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv


class FeatureDictionary(object):
    def __init__(self, trainfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        self.trainfile = trainfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        df = dfTrain
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique() # 这里估计是要做 one-hot一类操作， key 是原始值，value 是 转化为绝对坐标的值
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc

class DeepFMLearner(object):
    def __init__(self, env, eval_env, dfm_params, data_only):
        self.env = env
        self.dfm_params = dfm_params
        self.eval_env = eval_env
        self.data_only = data_only
        self._init()
        pass

    def _init(self):
        assert isinstance(self.env, MultiCityDriverEnv)
        assert isinstance(self.eval_env, MultiCityDriverEnv)
        self.datasets, feature_type_path = self.env.demer_data_gen(data_only=self.data_only)
        select_column = np.genfromtxt(feature_type_path, dtype='str', delimiter=' ')
        select_col_names = select_column.T[0]
        select_col_fillna = select_column.T[1]
        ignore_cols = []
        numeric_cols = []
        for feat_name, feat_type in zip(select_col_names, select_col_fillna):
            if feat_type == '0':
                ignore_cols.append(feat_name)
            elif feat_type == '1':
                numeric_cols.append(feat_name)
        ignore_cols.append("target")
        self.ignore_cols = ignore_cols
        self.numeric_cols = numeric_cols

    def learn(self, total_timesteps, log_interval):
        # initial dataset

        fd = FeatureDictionary(dfTrain=self.datasets,
                               numeric_cols=self.numeric_cols,
                               ignore_cols=self.ignore_cols)
        data_parser = DataParser(feat_dict=fd)
        Xi_train, Xv_train, y_train = data_parser.parse(df=self.datasets, has_label=True)

        self.dfm_params["feature_size"] = fd.feat_dim
        self.dfm_params["field_size"] = len(Xi_train[0])

        dfm = DeepFM(**self.dfm_params)
        # dfm.fit(Xi_train_, Xv_train_, y_train_)
        for epoch in range(dfm.epoch):
            time_step_holder.set_time(epoch)
            dfm.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / dfm.batch_size)
            losses = []
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = dfm.get_batch(Xi_train, Xv_train, y_train, dfm.batch_size, i)
                loss = dfm.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                losses.append(loss)
            logger.record_tabular("loss/mse_loss", np.mean(losses))

            if epoch % 5 == 0:
                obs = self.eval_env.reset(evaluation=EvaluationType.EVALUATION)
                for i in range(self.eval_env.days):
                    dataset = self.eval_env.gen_all_sa(obs)
                    Xi_test, Xv_test = data_parser.parse(df=dataset, has_label=False)
                    pred_fos = dfm.predict(Xi_test, Xv_test)
                    # reshape and argmax and do step
                    obs, rewards, dones, infos = self.eval_env.reshape_and_step(pred_fos)
                ep_info = infos['episode']
                keys = ep_info.keys()
                city = self.eval_env.city_list[0]
                for k in keys:
                    if k == 'city_name':
                        continue
                    if 'daily' in k:
                        continue
                    else:
                        logger.logkv('epi_info-{}/{}'.format(city, k), safe_mean([ep_info[k]]))
                expert_cp_eval_ep_infos = self.eval_env.selected_env.expert_info
                gmv_inc = (np.mean(ep_info['gmv']) - np.mean(expert_cp_eval_ep_infos['gmv'])) / (
                            np.mean(expert_cp_eval_ep_infos['gmv']) + 1e-6) * 100
                cost_inc = (np.mean(ep_info['cost']) - np.mean(expert_cp_eval_ep_infos['cost'])) / (
                            np.mean(expert_cp_eval_ep_infos['cost']) + 1e-6) * 100
                rew_inc = (np.mean(ep_info['real_rews']) - np.mean(expert_cp_eval_ep_infos['real_rews'])) / (
                            np.mean(expert_cp_eval_ep_infos['real_rews']) + 1e-6) * 100
                logger.logkv('ab_performance/ test-ab_gmv_inc_percent-{}'.format(city), gmv_inc)
                logger.logkv('performance/ test-rews-{}'.format(city), np.mean(ep_info['rews']))
                logger.logkv('ab_performance/ test-ab_cost_inc_percent-{}'.format(city), cost_inc)
                logger.logkv('ab_performance/ test-ab_rew_inc_percent-{}'.format(city), rew_inc)
            logger.dump_tabular()


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
