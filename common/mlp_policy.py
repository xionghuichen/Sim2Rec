import tensorflow as tf
import gym
import numpy as np

from tqdm import tqdm
# from zoopt import Dimension, Objective, Parameter, Opt

import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense
from common import logger
from common.tester import tester
from common.data_info import CouponActionInfo

def lambda_cases_construct(key, value):
    lambda_dict = {}
    assert len(key) == len(value)
    for i in range(len(key)):
        lambda_dict[key[i]] = eval("lambda : value[" + str(i) + "]", locals())
    return lambda_dict


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False,  *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.cac_space = None
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def config_dicretization(self, coupon_action_info=None, cluster_number=None, sample_number=None, iter_number=None,
                             tdgamma=None, lam=None, env=None, test_env=None,
                             find_high_frequency_coupon_info=None,
                             use_predict_q=False, use_predict_dist=False, use_value_weight=False, use_restore_act=False,
                             disc_number=100, disc_percent=0.9, disc_precision=2, use_revise_act=True,):
        self.coupon_action_info = coupon_action_info
        self.cluster_number = cluster_number
        self.sample_number = sample_number
        self.iter_number = iter_number
        self.tdgamma = tdgamma
        self.lam = lam
        self.env = env
        self.use_predict_q = use_predict_q
        self.test_env = test_env
        self.find_high_frequency_coupon_info = find_high_frequency_coupon_info
        self.use_predict_dist = use_predict_dist
        self.disc_number = disc_number
        self.disc_percent = disc_percent
        self.disc_precision = disc_precision
        self.use_value_weight =  use_value_weight
        self.use_restore_act = use_restore_act
        self.use_revise_act = use_revise_act
        self.driver_fos_mean = None
        self.driver_fos_std = None

        assert self.use_predict_dist + self.use_predict_q <= 1


    def _init(self, ob_space, ac_space, cac_space, hid_size, coupon_hid_layers, num_hid_layers, cms_min, dms_min,  cms_max, dms_max ,confounder, head_amount,
               tanh_oup,
              gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        fc_init_std_scale = 0.01
        std_range = 2.5 # TODO: is 2.5 too large as a scale of std.
        self.head_amount = head_amount
        self.cac_space = cac_space
        self.cac_min_value = cms_min
        self.dac_min_value = dms_min
        self.cac_max_value = cms_max
        self.dac_max_value = dms_max
        # self.ac_max_value = 6

        self.pdtype = pdtype = make_pdtype(ac_space)
        self.cpdtype = cpdtype = make_pdtype(cac_space)
        action_ph = pdtype.sample_placeholder([None, ])
        sequence_length = None

        self.ob = ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.stochastic = stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        self.pass_direct = pass_direct = U.get_placeholder(name="pass_direct", dtype=tf.bool, shape=())
        self.driver_type = U.get_placeholder(name="driver_type", dtype=tf.int32, shape=(sequence_length, 1))
        self.ca_ph = ca_ph = U.get_placeholder(name="ca", dtype=tf.float32, shape=[sequence_length, ] + list(cac_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        init_std_scale = 0.5
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -8.0, 8.0)
        # ---------------------policy_network----------------------
        # platform policy network
        pcoupon_out = obz
        for cp_i in range(coupon_hid_layers):
            pcoupon_out = tf.nn.tanh(dense(pcoupon_out, hid_size, "polfc_c{}".format(cp_i + 1), weight_init=U.normc_initializer(0.5)))

        pcoupon_out2 = pcoupon_out
        if gaussian_fixed_var and isinstance(cac_space, gym.spaces.Box):
            # cmean = tf.clip_by_value(
            #     dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "polfc_cout", U.normc_initializer(0.01)),
            #     self.cac_min_value, self.ac_max_value)
            if tanh_oup:
                cmean = (tf.nn.tanh(dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "polfc_cout",
                                          U.normc_initializer(init_std_scale))) + 1) * cms_max * 1.1 / 2 + cms_min
            else:
                # cmean = tf.clip_by_value(
                #     dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "polfc_cout", U.normc_initializer(fc_init_std_scale)),
                #     cms_min * 0.9, cms_max * 1.1)
                cmean = dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "polfc_cout", U.normc_initializer(fc_init_std_scale))
            clogstd = tf.get_variable(name="clogstd", shape=[1, cpdtype.param_shape()[0] // 2],
                                      initializer=tf.constant_initializer(0))
            cpdparam = tf.concat([cmean, cmean * 0.0 + clogstd], axis=1)
        else:
            if tanh_oup:
                cmean = (tf.nn.tanh(dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "cpolfinal_mean",
                                         U.normc_initializer(init_std_scale))) + 0.8) * (cms_max - cms_min) * 1.2 / 2 + cms_min
            else:
                cmean = dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "cpolfinal_mean", U.normc_initializer(fc_init_std_scale))
            clogstd = tf.log((tf.nn.tanh(dense(pcoupon_out2, cpdtype.param_shape()[0] // 2, "cpolfinal_std",
                                              U.normc_initializer(init_std_scale))) + 1.001) * std_range)
            cpdparam = tf.concat([cmean, cmean * 0.0 + clogstd], axis=1)
        self.cpd = cpdtype.pdfromflat(cpdparam)
        # platform policy action： cac
        cac = U.switch(stochastic, self.cpd.sample(), self.cpd.mode())
        cac_prob = [self.cpd.mean, self.cpd.std]
        ca = U.switch(pass_direct, cac, ca_ph)
        fca = tf.concat([ca[:, :-1], tf.sign(ca[:, -1:])], axis=1)
        if confounder:
            # hidden policy network ...
            hidden_inpt = tf.concat([obz, fca], axis=1)
            hid_pol_out1 = tf.nn.tanh(
                dense(hidden_inpt, hid_size, "hid_polfc_c1", weight_init=U.normc_initializer(0.5)))
            hid_pol_out2 = tf.nn.tanh(
                dense(hid_pol_out1, hid_size, "hid_polfc_c2", weight_init=U.normc_initializer(0.5)))

            hid_pol_out3 = dense(hid_pol_out2, cpdtype.param_shape()[0] // 2, "hid_polfc_cout",
                                 U.normc_initializer(0.01))
            last_out = tf.concat([obz, fca, hid_pol_out3], axis=1)
        else:

            last_out = tf.concat([obz, fca], axis=1)

        # driver policy network
        for i in range(num_hid_layers):
            if tanh_oup:
                last_out = tf.nn.tanh(
                    dense(last_out, hid_size, "polfc_d%i" % (i + 1), weight_init=U.normc_initializer(0.5)))
            else:
                last_out = tf.nn.leaky_relu(
                    dense(last_out, hid_size, "polfc_d%i" % (i + 1), weight_init=U.normc_initializer(0.5)))

        pd_values = []
        for i in range(head_amount):
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                # mean = tf.clip_by_value(
                #     dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_" + str(i), U.normc_initializer(0.01)),
                #     self.dac_min_value, self.ac_max_value)
                if tanh_oup:
                    mean = (tf.nn.tanh(dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_" + str(i), U.normc_initializer(0.01))) + 1) * dms_max * 1.1 / 2 + dms_min
                else:
                    mean = tf.clip_by_value(
                        dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_" + str(i), U.normc_initializer(fc_init_std_scale)),
                        dms_min * 0.9, dms_max * 1.1)
                logstd = tf.get_variable(name="logstd_" + str(i), shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                if tanh_oup:

                    mean = (tf.nn.tanh(dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_mean" + str(i),
                                             U.normc_initializer(init_std_scale))) + 0.8) * (dms_max - dms_min) * 1.2 / 2 + dms_min
                    # mean = (tf.nn.tanh(dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_mean" + str(i),
                    #                          U.normc_initializer(init_std_scale))) + 1) * dms_max * 1.1 / 2 + dms_min
                else:
                    # mean = tf.clip_by_value(
                    #     dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_mean" + str(i), U.normc_initializer(fc_init_std_scale)),
                    #     dms_min * 0.9, dms_max * 1.1)
                    mean = dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_mean" + str(i), U.normc_initializer(fc_init_std_scale))
                logstd = tf.log((tf.nn.tanh(dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_std" + str(i),
                                                   U.normc_initializer(init_std_scale))) + 1.001) * std_range)
                # logstd = tf.log((tf.nn.tanh(dense(last_out, pdtype.param_shape()[0] // 2, "polfinal_std" + str(i),
                #                          U.normc_initializer(0.01))) + 1) * (dms_max -dms_min) * 1.1 / 2)
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            pd_values.append(tf.expand_dims(pdparam, 1))
        multi_pd_param = tf.concat(pd_values, axis=1)
        pdparam = tf.squeeze(tf.batch_gather(multi_pd_param, self.driver_type), axis=1)
        self.pd = pdtype.pdfromflat(pdparam)
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        # -----------------------v_network-------------------------
        # platform v_network [coupon v-network]
        coupon_out1 = tf.nn.tanh(dense(obz, hid_size, "vffc_c1", weight_init=U.normc_initializer(0.5)))
        coupon_out2 = tf.nn.tanh(dense(coupon_out1, hid_size, "vffc_c2", weight_init=U.normc_initializer(0.5)))
        self.cvpred = dense(coupon_out2, 1, "vffc_cout", weight_init=U.normc_initializer(1.0))[:, 0]
        # paltform coupon q_network
        fca = tf.concat([ca_ph[:, :-1], tf.sign(ca_ph[:, -1:])], axis=1)
        q_inpt = tf.concat([obz, fca], axis=1)
        coupon_out1 = tf.nn.tanh(dense(q_inpt, hid_size, "qffc_c1", weight_init=U.normc_initializer(0.5)))
        coupon_out2 = tf.nn.tanh(dense(coupon_out1, hid_size, "qffc_c2", weight_init=U.normc_initializer(0.5)))
        self.cq_pred = dense(coupon_out2, 1, "qffc_cout", weight_init=U.normc_initializer(1.0))[:, 0]

        if confounder:
            # hidden policy network ...
            hidden_inpt = tf.concat([obz, fca], axis=1)
            hid_pol_out1 = tf.nn.tanh(
                dense(hidden_inpt, hid_size, "hid_polfc_c1", weight_init=U.normc_initializer(0.5), reuse=True))
            hid_pol_out2 = tf.nn.tanh(
                dense(hid_pol_out1, hid_size, "hid_polfc_c2", weight_init=U.normc_initializer(0.5), reuse=True))

            hid_pol_out3 = dense(hid_pol_out2, cpdtype.param_shape()[0] // 2, "hid_polfc_cout",
                                 U.normc_initializer(0.01), reuse=True)

            # driver policy network
            last_out = tf.concat([obz, fca, hid_pol_out3], axis=1)
        else:
            # driver policy network
            last_out = tf.concat([obz, fca], axis=1)
        # value network
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc_d%i" % (i + 1), weight_init=U.normc_initializer(0.5)))
        vpred_values = []
        for i in range(head_amount):
            vpred = dense(last_out, 1, "vffc_dout_" + str(i), weight_init=U.normc_initializer(1.0))[:, 0]
            vpred_values.append(tf.expand_dims(vpred, 1))
        # for i in range(len(vpred_keys)):
        #     vpred_cases[vpred_keys[i]] = eval("lambda : vpred_values[" + str(i) + "]")
        multi_vpred_param = tf.concat(vpred_values, axis=1)
        pdparam = tf.squeeze(tf.batch_gather(multi_vpred_param, self.driver_type), axis=1)
        self.vpred = pdparam  # tf.case(vpred_cases, default=default_error, exclusive=True)
        self._calcll = U.function([action_ph, ob, self.driver_type], [self.pd.logp(action_ph)])
        self._call_cvpred = U.function([ob], [self.cvpred])
        self._call_cq_pred = U.function([ob, ca_ph], [self.cq_pred])
        self.cac = cac
        self._cac_prob = U.function([ob], [cac_prob])
        self._cact = U.function([stochastic, ob], [self.cac, self.cvpred])
        self.dac = ac
        self._act = U.function([stochastic, ob, ca_ph, self.driver_type, pass_direct], [self.dac, self.vpred],
                               givens={pass_direct: False})
        self._both_act = U.function([stochastic, ob, ca_ph, self.driver_type, pass_direct],
                                    [self.cac, self.cvpred, self.dac, self.vpred],
                                    givens={ca_ph: tf.zeros(shape=tf.shape(cac)),
                                            pass_direct: True})

    def both_act(self, stochastic, ob, driver_type):
        if len(ob.shape) == 1:
            fake_cac_inpt = np.zeros(shape=[1] + list(self.cac_space.shape))
            cac, cvpred, dac, dvpred = self._both_act(stochastic, ob[None], fake_cac_inpt, [[driver_type]])
            return cac[0], cvpred[0], dac[0], dvpred[0]
        else:
            fake_cac_inpt = np.zeros(shape=[ob.shape[0]] + list(self.cac_space.shape))
            cac, cvpred, dac, dvpred = self._both_act(stochastic, ob, fake_cac_inpt, driver_type)
            return cac, cvpred, dac, dvpred

    def cact_prob(self, ob):
        if len(ob.shape) == 1:
            cac = self._cac_prob(ob[None])
            return cac[0]
        else:
            cac = self._cac_prob(ob)
            return cac

    def cact(self, stochastic, ob, driver_type=None):
        if len(ob.shape) == 1:
            cac, cvpred = self._cact(stochastic, ob[None])
            return cac[0], cvpred[0]
        else:
            cac, cvpred = self._cact(stochastic, ob)
            return cac, cvpred

    def act(self, stochastic, ob, ca, driver_type):
        if len(ob.shape) == 1:
            dac, vpred = self._act(stochastic, ob[None], ca[None], [[driver_type]])
            return dac[0], vpred[0]
        else:
            if len(driver_type.shape) == 1:
                driver_type = np.expand_dims(driver_type, axis=1)
            dac, vpred = self._act(stochastic, ob, ca, driver_type)
            return dac, vpred

    def calcll(self, action, ob, driver_type):
        logliks = self._calcll(action, ob[None], [[driver_type]])
        return logliks[0][0]

    def call_cvpred(self, ob):
        if len(ob.shape) == 1:
            vpred = self._call_cvpred(ob[None])
            return vpred[0]
        else:
            vpred = self._call_cvpred(ob)
            return vpred[0]

    def call_cqpred(self, ob, ac):
        if len(ob.shape) == 1:
            vpred = self._call_cq_pred(ob[None], ac)
            return vpred[0]
        else:
            vpred = self._call_cq_pred(ob, ac)
            return vpred[0]


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    #### 以下是线上测试才会用到的代码

    def compute_q_value_rollout(self, obs, actions, driver_index=None):
        from driver_simenv_disc import Driver_Env
        assert isinstance(self.env, Driver_Env) and isinstance(self.test_env, Driver_Env)
        next_obs, reward, done, info = self.test_env.rollout(obs, actions, driver_index, source_pi=self.env.pi)
        vpred = self.call_cvpred(next_obs)
        if done:
            q_value = reward
        else:
            q_value = reward + self.tdgamma * self.lam * vpred

        return q_value, info['coupon_info']

    def compute_q_value_l2_dist(self, obs, actions, driver_index=None, driver_types=None):
        real_acs, value = self.cact(False, obs, driver_types)
        if self.use_revise_act:
            real_acs = self.env.restore_coupon_info(real_acs, reshape_sec_fos=False)
            real_acs = self.revise_cact(real_acs)
            real_acs = self.coupon_action_info.construct_coupon_info(real_acs, sec_reshape=False, is_r_factor=True)
        if self.use_restore_act:
            real_acs = self.env.restore_coupon_info(real_acs, reshape_sec_fos=False)
        if self.use_value_weight:
            return -1 * np.mean(np.square(actions - real_acs), axis=1) * value / np.sum(value), None
        else:
            return - 1 * np.mean(np.square(actions - real_acs), axis=1), None


    def compute_q_value_predict(self, obs, actions, driver_index=None):
        from driver_simenv_disc import Driver_Env
        assert isinstance(self.env, Driver_Env) and isinstance(self.test_env, Driver_Env)
        # next_obs, reward, done, info = self.test_env.rollout(obs, actions, driver_index, source_pi=self.env.pi)
        q_value = self.call_cqpred(obs, actions)
        return q_value, None # , info['coupon_info']

    def valus_and_max_in_actions(self, obs, driver_types, discreted_actions, driver_index=None):
        q_values = np.zeros(shape=(discreted_actions.shape[0], obs.shape[0]))
        for idx, da in enumerate(discreted_actions):
            das = np.repeat([da], obs.shape[0], axis=0)
            if self.use_predict_q:
                q_value, _ = self.compute_q_value_predict(obs, das, driver_index)
            elif self.use_predict_dist:
                q_value, _ = self.compute_q_value_l2_dist(obs, das, driver_index, driver_types)
            else:
                q_value, _ = self.compute_q_value_rollout(obs, das, driver_index)
            # vpred = self.call_cvpred(obs)
            q_values[idx] = q_value
        max_value_action = np.argmax(q_values, axis=0)
        return q_values, max_value_action

    def mean_action_value(self, obs, cluster_driver_idx, driver_types, actions):
        v_preds, max_value_action = self.valus_and_max_in_actions(obs[cluster_driver_idx],
                                                   driver_types[cluster_driver_idx], actions, cluster_driver_idx)
        return np.mean(v_preds, axis=1)
        # max_value = np.max(v_preds, axis=0)
        # return self.calculate_mean_value_of_cluster(max_value_action, max_value, cluster_number=actions.shape[0])

    def calculate_mean_value_of_cluster(self, max_value_action, max_value, cluster_number):
        mean_value_of_cluster = np.zeros(shape=(cluster_number))
        for cn in range(cluster_number):
            if np.where(max_value_action == cn)[0].shape[0] == 0:
                # empty cluster
                mean_value_of_cluster[cn] = 0
            else:
                mean_value_of_cluster[cn] = np.mean(max_value[np.where(max_value_action == cn)])
        return mean_value_of_cluster

    def reset_empty_cluster(self, discreted_actions, mean_value_of_cluster):
        empty_cluster = np.where(mean_value_of_cluster == 0)
        # max_mean_value = np.max(mean_value_of_cluster)
        max_value_index = np.argmax(mean_value_of_cluster)
        mean_value_of_cluster[empty_cluster] = 0 # max_mean_value
        discreted_actions[empty_cluster] = discreted_actions[max_value_index]

    def evaluate_func(self, solution, obs, target_value):
        from driver_simenv_disc import Driver_Env
        assert isinstance(self.env, Driver_Env)
        assert isinstance(self.coupon_action_info, CouponActionInfo)
        actions = solution.get_x()
        actions = np.array(actions).reshape(self.cluster_number, self.env.dim_coupon_feature)
        reshape_actions = self.coupon_action_info.construct_coupon_info(actions)
        q_values = np.zeros(shape=(self.cluster_number, obs.shape[0]))
        for idx, da in enumerate(reshape_actions):
            das = np.repeat([da], obs.shape[0], axis=0)
            q_value, _ = self.compute_q_value(obs, das)
            # vpred = self.call_cvpred(obs)
            q_values[idx] = q_value
        max_value = np.max(q_values, axis=0)
        max_value_action = np.argmax(q_values, axis=0)
        solution.set_post_attach(max_value_action)
        # q_value = self.compute_q_value(obs, reshape_actions)
        return target_value - np.mean(max_value)

    def revise_cact(self, restore_acs, fo_reshape=False):
        # revise_df['coupon_param_3'] = revise_df['coupon_param_3'] - revise_df['coupon_param_1']
        # 一级补贴要大于等于历史完单量的1.3倍
        def zero_filter(cond, axis):
            return np.logical_and(cond, restore_acs[:, axis] > 0)
        if fo_reshape:
            select_index = np.where(np.isin(self.env.driver_ids, self.env.select_dids))[0]

            logger.info("select did {}, shape {}".format(self.env.select_dids, self.env.select_dids.shape))
            logger.info("select index {}".format(select_index))
            fos_mean = self.driver_fos_mean[select_index]
            fos_std = self.driver_fos_std[select_index]
            restore_acs = np.copy(restore_acs)
            assert isinstance(self.coupon_action_info, CouponActionInfo)
            c1_min = restore_acs[:, 0] - fos_mean * 1.3
            id_reshape = np.where(zero_filter(c1_min < 0, 0))[0]
            restore_acs[np.where(zero_filter(c1_min < 0, 0)), 0] = fos_mean[np.where(zero_filter(c1_min < 0, 0))] * 1.3
            # revise_df['coupon_param_1'][c1_min < 0] = revise_df['fos_mean'][c1_min < 0] * 1.3
            fos_sup = fos_mean + fos_std * 1.3
            # revise_df['fos_sup'] = revise_df['fos_mean'] + revise_df['fos_std'] * 1.3
            # 一级补贴门槛要大于等于  历史完单均值 + 标准差 * 1.3
            c1_max = restore_acs[:, 0] - fos_sup * 1.3
            id_reshape_append = np.where(zero_filter(c1_max < 0, 0))[0]
            id_reshape = np.append(id_reshape, id_reshape_append)
            # c1_max = revise_df['coupon_param_1'] - revise_df['fos_sup']
            restore_acs[np.where(zero_filter(c1_max < 0, 0)), 0] = fos_sup[np.where(zero_filter(c1_max < 0, 0))] * 1.3
            # revise_df['coupon_param_1'][c1_max < 0] = revise_df['fos_sup'][c1_max < 0]
            # revise_df['coupon_param_1'][revise_df['coupon_param_1']<7] = 7

            id_reshape = np.append(id_reshape, np.where(restore_acs[:, 0] > 29)[0])
            # revise_df['coupon_param_1'][revise_df['coupon_param_1'] > 29] = 29
            # 二级补贴：要在历史完单量的标准差的2 到2.5倍区间
            c3_min = restore_acs[:, 2] - 2.0 * fos_std
            c3_max = restore_acs[:, 2] - 2.5 * fos_std
            # c3_min = revise_df['coupon_param_3'] - 2.0 * revise_df['fos_std']
            # c3_max = revise_df['coupon_param_3'] - 2.5 * revise_df['fos_std']
            restore_acs[np.where(zero_filter(c3_min < 0, 2)), 2] = fos_std[np.where(zero_filter(c3_min < 0, 2))] * 2.0
            restore_acs[np.where(zero_filter(c3_max < 0, 2)), 2] = fos_std[np.where(zero_filter(c3_max < 0, 2))] * 2.5

            # id_reshape = np.append(id_reshape, np.where(c3_min < 0)[0])
            # id_reshape = np.append(id_reshape, np.where(c3_max < 0)[0])
            # id_reshape = np.append(id_reshape, np.where(restore_acs[:, 2] > 20)[0])
            # revise_df['coupon_param_3'][c3_min < 0] = revise_df['fos_std'][c3_min < 0] * 2.0
            # revise_df['coupon_param_3'][c3_max > 0] = revise_df['fos_std'][c3_max > 0] * 2.5
            # revise_df['coupon_param_3'][revise_df['coupon_param_3']<4]=4
            # revise_df['coupon_param_3'][revise_df['coupon_param_3'] > 20] = 20
        restore_acs[np.where(restore_acs[:, 0] > 29), 0] = 25
        restore_acs[np.where(restore_acs[:, 2] > 20), 2] = 20
        # 下面是单均补贴 约束，根据历史单均补贴的均值和方差来做限制的
        mean1 = self.coupon_action_info.coupon_mean[1]
        std1 = self.coupon_action_info.coupon_std[1]
        c2_min = mean1 - 0.6 * std1
        c2_max = mean1 - 0.4 * std1

        restore_acs[np.where(zero_filter(restore_acs[:, 1] < c2_min, 1)), 1] = c2_min
        restore_acs[np.where(zero_filter(restore_acs[:, 1] < c2_max, 1)), 1] = c2_max
        restore_acs[np.where(zero_filter(restore_acs[:, 1] < 0.9, 1)), 1] = 0.9
        if not fo_reshape:
            id_reshape = np.where(restore_acs[:, 1] < 0.9)[0]
        else:
            id_reshape = np.append(id_reshape, np.where(restore_acs[:, 1] < 0.9)[0])
        id_reshape = np.append(id_reshape, np.where(restore_acs[:, 1] < c2_min)[0])
        id_reshape = np.append(id_reshape, np.where(restore_acs[:, 1] > c2_max)[0])
        # restore_acs['coupon_param_2'][revise_df['coupon_param_2'] < c2_min] = c2_min
        # revise_df['coupon_param_2'][revise_df['coupon_param_2'] > c2_max] = c2_max
        # revise_df['coupon_param_2'][revise_df['coupon_param_2'] < 0.9] = 0.9

        mean2 = self.coupon_action_info.coupon_mean[3]
        std2 = self.coupon_action_info.coupon_std[3]
        # mean2 = coupon_mean_std[3][0]
        # std2 = coupon_mean_std[3][1]
        c4_min = mean1 - 0.6 * std1 + mean2
        c4_max = mean1 + mean2 + 0.4 * std2
        restore_acs[np.where(zero_filter(restore_acs[:, 3] < c4_min, 3)), 3] = c4_min
        restore_acs[np.where(zero_filter(restore_acs[:, 3] > c4_max, 3)), 3] = c4_max

        id_reshape = np.append(id_reshape, np.where(restore_acs[:, 3] < c4_min)[0])
        id_reshape = np.append(id_reshape, np.where(restore_acs[:, 3] > c4_max)[0])
        id_reshape = np.unique(id_reshape)
        import random
        # if random.randint(1, 1000) == 500:
        #     logger.info("revise percent : {}".format(float(id_reshape.shape[0])/select_index.shape[0]))
        # revise_df['coupon_param_4'][revise_df['coupon_param_4'] < c4_min] = c4_min
        # revise_df['coupon_param_4'][revise_df['coupon_param_4'] > c4_max] = c4_max
        restore_acs[:, 0] = np.round(restore_acs[:, 0], 0)
        restore_acs[:, 2] = np.round(restore_acs[:, 2], 0)
        return restore_acs


    def discretization_c_act_high_rew(self, stochastic, obs, driver_types):
        np.set_printoptions(precision=3, suppress=True)
        hf_cac = self.find_high_frequency_coupon_info # sec_reshape=False
        # restore -> revise -> construct
        # if self.use_revise_act:
        #     hf_cac = self.revise_cact(hf_cac)
        reshape_cac = self.coupon_action_info.construct_coupon_info(hf_cac, is_r_factor=True, sec_reshape=False)
        if self.use_restore_act:
            reshape_cac = hf_cac
        q_value, max_value_action = self.valus_and_max_in_actions(obs, driver_types, discreted_actions=reshape_cac)
        max_value = np.max(q_value, axis=0)
        # self.call_cqpred(obs[5], self.cact(False, obs[1], driver_types[1])[0]) - max_value[5]
        to_remove_index = []
        selected_index = np.unique(max_value_action)
        if np.unique(max_value_action).shape[0] > self.cluster_number:
            for i in range(reshape_cac.shape[0]):
                if i not in selected_index:
                    to_remove_index.append(i)
            to_remove_index = np.array(to_remove_index)
            logger.info("drop empty acion {}".format(to_remove_index))
            sub_reshape_cac = reshape_cac[selected_index]
        else:
            sub_reshape_cac = reshape_cac[selected_index]
            # reshape_cac = reshape_cac[left_index]
        bin_count = np.bincount(max_value_action)
        # bin_count_gt_0_index = np.where(bin_count > 0)
        bin_count_gt_0 = bin_count[np.where(bin_count > 0)]
        bin_count_index = np.where(bin_count > 0)
        while sub_reshape_cac.shape[0] > self.cluster_number:
            # min_index_in_count = np.argmin(bin_count_gt_0)
            index_in_count_min = 0
            min_reward_loss = 10000
            max_value_action_i_min = None
            min_index_driver_min = None
            q_losses = []
            for count_index in tqdm(range(bin_count_gt_0.shape[0])):
                q_value_i, max_value_action_i = self.re_allocate(count_index, selected_index, q_value)

                q_loss = np.sum(max_value) - np.sum(np.max(q_value_i, axis=0))
                min_index_driver = np.where(max_value_action == selected_index[count_index])[0]
                logger.info(
                    "try: drop min: count {}, amount {}. loss {}".format(count_index,
                                                                           min_index_driver.shape[0], q_loss))

                if np.all(q_loss < np.array(q_losses)):
                    index_in_count_min = count_index
                    min_reward_loss = q_loss
                    max_value_action_i_min = max_value_action_i
                    min_index_driver_min = min_index_driver
                q_losses.append(q_loss)
            logger.info("q_losses {}".format(q_losses))
            selected_index = np.delete(selected_index, index_in_count_min)
            logger.info("left {}. drop min: index {}, count {} loss {}".format(sub_reshape_cac.shape[0], index_in_count_min,
                        min_index_driver_min.shape[0], min_reward_loss))
            sub_reshape_cac = reshape_cac[selected_index]
            bin_count = np.bincount(max_value_action_i_min)
            bin_count_gt_0 = bin_count[np.where(bin_count > 0)]
            # max_value_action = max_value_action_i_min
            logger.info("bin_count_gt_0 {}, amount {}".format(bin_count_gt_0, np.sum(bin_count_gt_0)))

        # for last iteration.
        v_preds, cluster_of_driver = self.valus_and_max_in_actions(obs, driver_types, sub_reshape_cac)

        bin_count_final = np.bincount(cluster_of_driver)
        bin_count_gt_0_2 = bin_count_final[np.where(bin_count_final > 0)]
        logger.info("[final]q loss {}".format(np.sum(max_value) - np.sum(np.max(v_preds, axis=0))))
        logger.info("[final] bin_count_gt_0 {},  amount {}".format(bin_count_gt_0_2, np.sum(bin_count_gt_0_2)))
        logger.info("[iter-final] bin_count_gt_0 {},  amount {}".format(bin_count_gt_0, np.sum(bin_count_gt_0)))
        logger.info("--- print discretization cact -----")
        selected_index = np.unique(cluster_of_driver)
        for i in selected_index:
            match_index = np.where(np.all(reshape_cac == sub_reshape_cac[i], axis=1))[0]
            if self.use_restore_act:
                restore_cp = sub_reshape_cac[i]
            else:
                restore_cp = self.env.restore_coupon_info([sub_reshape_cac[i]], reshape_sec_fos=False)
            # restore_cp[:, 2:4] = restore_cp[:, 2:4] - restore_cp[:, 0:2]
            logger.info(
                "index :{}, count: {}, coupon: {}. original cp: {}".format(match_index, np.where(cluster_of_driver == i)[0].shape[0],
                                                          restore_cp, sub_reshape_cac[i]))
            assert np.all(hf_cac[match_index, :] - restore_cp < 0.01), "original {}, restore {}".format(
                hf_cac[match_index, :], restore_cp)

        logger.info("--- print discretization cact -----")
        # TODO not pass this assert yet
        # assert np.all(bin_count_gt_0_2 == bin_count_gt_0)
        return sub_reshape_cac[cluster_of_driver], None

    def re_allocate(self, count_index, selected_index, total_q_value):
        new_selected_index = np.delete(selected_index, count_index)
        q_value_i = total_q_value[ new_selected_index, :]
        max_value_action_i = np.argmax(q_value_i, axis=0)
        # q_value_i, max_value_action_i = self.valus_and_max_in_actions(obs[min_index_driver],
        #                                                               driver_types[min_index_driver],
        #                                                               discreted_actions=sub_reshape_cac,
        #                                                               driver_index=min_index_driver)
        return q_value_i, max_value_action_i