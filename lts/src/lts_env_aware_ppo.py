import time
import sys
from collections import deque

from abc import ABC, abstractmethod

import gym
import numpy as np
import tensorflow as tf

from common import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from ppo2.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from transfer_learning.src.policies import EnvAwarePolicy, EnvExtractorPolicy
from transfer_learning.src.driver_simenv_disc import MultiCityDriverEnv
from vae.src.vae_cls_fix import VaeHander
from common.tester import tester
from common.utils import *
from common.config import *
from common.mpi_running_mean_std import RunningMeanStd
from common.emb_unit_test import emb_unit_test_cont
from common.tf_func import *
from lts.src.experiment_env import EnvAwareRunner as Runner

def get_target_updates(_vars, target_vars, tau, verbose=0):
    """
    get target update operations

    :param _vars: ([TensorFlow Tensor]) the initial variables
    :param target_vars: ([TensorFlow Tensor]) the target variables
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :return: (TensorFlow Operation, TensorFlow Operation) initial update, soft update
    """
    if verbose >= 2:
        logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)
    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)
    return tf.group(*init_updates), tf.group(*soft_updates)

class PPO2(ActorCriticRLModel):
    TRAINING_POLICY = 'tp'
    TRAINING_LSTM = 'tl'
    TRAINING_ALL ='ta'

    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, sess, policy, env_extractor_policy, distribution_embedding,
                 hand_code_distribution, env, eval_env, all_domain_list,
                 env_params_size=20, gamma=0.99, n_steps=128, ent_coef=0.01,
                 v_learning_rate=2.5e-4, p_learning_rate=2.5e-4, p_env_learning_rate=2.5e-4, vf_coef=0.5,
                 p_grad_norm=0.5, v_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None, driver_cluster_days=7,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, env_extractor_policy_kwargs=None,
                 stable_dist_embd=False, l2_reg_pi=0, l2_reg_env=0, l2_reg_v=0, tau=0.001,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, name='ppo', use_lda_loss=False, log_lda=False,
                 consistent_ecoff=0, lda_max_grad_norm=-1, constraint_lda=True, cliped_lda=False,
                 square_lda=False,
                 record_gradient=False, just_scale=False, normalize_rew=False, scaled_lda=False,
                 lstm_train_freq=-1, rms_opt=False, remove_done=False, soft_update_freq=1, log_interval=10):

        super(PPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                   seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.lstm_train_freq = lstm_train_freq
        self.rms_opt = rms_opt
        self.tau = tau
        self.soft_update_freq = soft_update_freq
        self.all_domain_list = all_domain_list
        self.remove_done = remove_done
        self.log_interval = log_interval

        self.cliped_lda =cliped_lda
        self.square_lda = square_lda
        self.scaled_lda = scaled_lda
        self.record_gradient = record_gradient
        self.v_learning_rate = v_learning_rate
        self.p_learning_rate = p_learning_rate
        self.p_env_learning_rate = p_env_learning_rate
        self.l2_reg_pi = l2_reg_pi
        self.l2_reg_env = l2_reg_env
        self.l2_reg_v = l2_reg_v
        self.just_scale = just_scale
        self.normalize_rew = normalize_rew
        self.stable_dist_embd = stable_dist_embd
        self.distribution_dict = {}
        self.env_params_size = env_params_size
        self.cliprange = cliprange
        self.eval_env = eval_env
        self.log_lda = log_lda
        self.env_extractor_policy = env_extractor_policy
        self.hand_code_distribution = hand_code_distribution
        self.distribution_embedding = distribution_embedding
        self.use_lda_loss = use_lda_loss
        self.driver_cluster_days = driver_cluster_days
        self.lda_max_grad_norm = lda_max_grad_norm
        self.constraint_lda = constraint_lda
        self.cliprange_vf = cliprange_vf
        self.name = name
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.p_grad_norm = p_grad_norm
        self.v_grad_norm = v_grad_norm
        self.normalize_rew = normalize_rew
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs

        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.consistent_ecoff = consistent_ecoff
        self.env_extractor_policy_kwargs = {} if env_extractor_policy_kwargs is None else env_extractor_policy_kwargs

        self.graph = None
        self.sess = sess
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.proba_step = None
        self.initial_state = None
        self.n_batch = None
        self.n_train_domains = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_model()


    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    @property
    def use_distribution_embedding(self):
        return self.distribution_embedding is not None

    @property
    def default_distribution_embedding(self):
        return np.zeros(shape=[10], dtype=np.float32)

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, EnvAwarePolicy), "Error: the input policy for the PPO2 model must be " \
                                                            "an instance of common.policies.ActorCriticPolicy."
            assert issubclass(self.env_extractor_policy, EnvExtractorPolicy)
            assert (not self.use_distribution_embedding) or isinstance(self.distribution_embedding, VaeHander)
            # self.nminibatches = self.n_envs
            # n_batch: 训练时采用的 batch size的组数，他是 n_env * n_steps 的倍数 （即每n_env * n_steps 为一组）
            self.n_train_domains = len(self.env.domain_list)
            assert self.n_envs % self.nminibatches == 0, "For recurrent policies, " \
                                                         "the number of environments run in parallel should be a multiple of nminibatches."
            self.n_batch = int(self.n_envs * self.n_steps * self.n_train_domains)

            self.graph = tf.get_default_graph()
            # env extractor info
            env_extractor_obs_space = box_concate(self.observation_space, self.action_space)
            env_extractor_obs_space = box_concate(env_extractor_obs_space, self.observation_space)
            identity = np.ones(shape=(self.env_params_size,))
            env_extractor_acs_space = gym.spaces.Box(low=identity * -1, high=identity, dtype=np.float32, )
            with self.graph.as_default():
                with tf.variable_scope(self.name):
                    if self.normalize_rew:
                        with tf.variable_scope('rew_rms'):
                            self.rew_rms = RunningMeanStd()
                    else:
                        self.rew_rms = None
                    if self.use_distribution_embedding:
                        dist_shape = self.env.representative_city.env_z_info.shape
                    else:
                        dist_shape = self.default_distribution_embedding.shape
                    self.distribution_emb_info = distribution_emb_info = tf.placeholder(tf.float32, [None] + list(dist_shape), name="distribution_emb_ph")
                    self.set_random_seed(self.seed)
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches
                    env_extractor_act_model = self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                        env_extractor_acs_space, self.n_envs, 1,
                                                                        n_batch_step,
                                                                        consistent_ecoff=self.consistent_ecoff,
                                                                        state_distribution_emb=distribution_emb_info,
                                                                        reuse=False, **self.env_extractor_policy_kwargs)
                    self.old_env_extractor_act_model = old_env_extractor_act_model = \
                        self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                        env_extractor_acs_space, self.n_envs, 1,
                                                                        n_batch_step,
                                                                        consistent_ecoff=self.consistent_ecoff,
                                                                        state_distribution_emb=distribution_emb_info,
                                                                        name='old_env_extractor',
                                                                        reuse=False, **self.env_extractor_policy_kwargs)

                    self.env_extractor_eval_dict = {}
                    self.old_env_extractor_eval_dict = {}
                    self.policy_eval_dict = {}
                    self.env_param_ph = tf.placeholder(tf.float32, [None, self.env_params_size], name="env_params")
                    self.old_env_param_ph = tf.placeholder(tf.float32, [None, self.env_params_size], name="old_env_params")
                    act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                            n_batch_step, env_parameter_input=self.env_param_ph,
                                            old_env_parameter_input=self.old_env_param_ph,
                                            state_distribution_emb=distribution_emb_info,
                                            reuse=False, **self.policy_kwargs)

                    self.old_env_extractor_train_model = old_env_extractor_train_model = self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                              env_extractor_acs_space,
                                                                              self.n_envs * self.n_train_domains,
                                                                              self.n_steps,
                                                                              n_batch_train,
                                                                              consistent_ecoff=self.consistent_ecoff,
                                                                              name='old_env_extractor',
                                                                              state_distribution_emb=distribution_emb_info,
                                                                              reuse=True,
                                                                              **self.env_extractor_policy_kwargs)
                    for domain in self.all_domain_list:
                        n_batch_step = driver_number = self.env.env_dict[domain].num_envs
                        # logger.info("city {}, driver ids {}".format(city, n_batch_step))
                        env_extractor_eval_model = self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                            env_extractor_acs_space, driver_number, 1,
                                                                            n_batch_step,
                                                                            consistent_ecoff=self.consistent_ecoff,
                                                                            name='env_extractor',
                                                                            state_distribution_emb=distribution_emb_info,
                                                                            reuse=True, **self.env_extractor_policy_kwargs)
                        eval_model = self.policy(self.sess, self.observation_space, self.action_space, driver_number, 1,
                                                 n_batch_step, env_parameter_input=self.env_param_ph,
                                                 old_env_parameter_input=self.old_env_param_ph,
                                                 state_distribution_emb=distribution_emb_info,
                                                 reuse=True, **self.policy_kwargs)

                        self.env_extractor_eval_dict[domain] = env_extractor_eval_model
                        self.policy_eval_dict[domain] = eval_model
                    for domain in self.all_domain_list:
                        n_batch_step = driver_number = self.env.env_dict[domain].num_envs
                        # logger.info("city {}, driver ids {}".format(city, n_batch_step))
                        old_env_extractor_eval_model = self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                            env_extractor_acs_space, driver_number, 1,
                                                                            n_batch_step,
                                                                            consistent_ecoff=self.consistent_ecoff,
                                                                            name='old_env_extractor',
                                                                            state_distribution_emb=distribution_emb_info,
                                                                            reuse=True, **self.env_extractor_policy_kwargs)
                        self.old_env_extractor_eval_dict[domain] = old_env_extractor_eval_model
                    with tf.variable_scope("train_model", reuse=True,
                                           custom_getter=tf_util.outer_scope_getter("train_model")):
                        env_extractor_train_model = self.env_extractor_policy(self.sess, env_extractor_obs_space,
                                                                              env_extractor_acs_space, self.n_envs * self.n_train_domains, self.n_steps,
                                                                              n_batch_train,
                                                                              consistent_ecoff=self.consistent_ecoff,
                                                                              state_distribution_emb=distribution_emb_info,
                                                                              reuse=True, **self.env_extractor_policy_kwargs)

                        self.driver_type_param = driver_type_param = env_extractor_train_model.action
                        old_driver_type_param = old_env_extractor_train_model.action

                        train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs * self.n_train_domains,
                                                  self.n_steps, n_batch_train,
                                                  old_env_parameter_input=old_driver_type_param,
                                                  env_parameter_input=driver_type_param,
                                                  state_distribution_emb=distribution_emb_info,
                                                  reuse=True, **self.policy_kwargs)
                        value_hidden_state = train_model
                        train_model_fix_env = self.policy(self.sess, self.observation_space, self.action_space,
                                                  self.n_envs * self.n_train_domains,
                                                  self.n_steps, n_batch_train,
                                                  env_parameter_input=tf.stop_gradient(driver_type_param),
                                                  old_env_parameter_input=old_driver_type_param,
                                                  state_distribution_emb=distribution_emb_info,
                                                  processed_obs=train_model.processed_obs,
                                                  reuse=True, **self.policy_kwargs)

                    self.train_model = train_model
                    self.train_model_fix_env = train_model_fix_env
                    self.act_model = act_model
                    self.env_extractor_act_model = env_extractor_act_model
                    self.env_extractor_train_model = env_extractor_train_model
                    # self.step = act_model.step
                    self.proba_step = act_model.proba_step
                    # self.value = act_model.value
                    self.initial_state = env_extractor_act_model.initial_state
                    self.policy_initial_state = act_model.initial_state
                    with tf.variable_scope("loss", reuse=False):
                        self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                        self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                        self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                        self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                        self.v_learning_rate_ph = tf.placeholder(tf.float32, [], name="v_learning_rate_ph")
                        self.p_learning_rate_ph = tf.placeholder(tf.float32, [], name="p_learning_rate_ph")
                        self.p_env_learning_rate_ph = tf.placeholder(tf.float32, [], name="p_env_learning_rate_ph")
                        self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                        neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                        neglogpac_fix_env = train_model_fix_env.proba_distribution.neglogp(self.action_ph)
                        self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                        self.entropy_fix_env = tf.reduce_mean(train_model_fix_env.proba_distribution.entropy())

                        vpred = train_model.value_flat
                        # soft update-env extractor
                        self.init_updates, self.soft_updates = get_target_updates(tf_util.get_trainable_vars(self.name + '/' + env_extractor_act_model.name),
                                                                        tf_util.get_trainable_vars(self.name + '/' + old_env_extractor_train_model.name), self.tau, 2)


                        # compute  LDA loss
                        self.driver_param_cluster_ph = tf.placeholder(tf.int32, [None, None], name="driver_param_cluster_ph")
                        i0 = tf.constant(0)
                        m0 = tf.constant(0.0)
                        m_bet = tf.zeros(shape=(1, self.env_params_size))
                        _between_condition = lambda loop_var, m: tf.less(loop_var, tf.shape(self.driver_param_cluster_ph)[0])
                        mu = tf.reduce_mean(driver_type_param, axis=0)
                        # TODO: check the shape
                        # assert self.nminibatches == 1, "nminibatches -> feed all of drivers"
                        driver_type_param_seq = tf.reshape(driver_type_param, shape=(-1, self.n_steps, driver_type_param.shape[-1]))
                        # every time-steps has a type cluster.
                        def _between_body(loop_var, mui_list):
                            cluster_mask_filter = tf.boolean_mask(self.driver_param_cluster_ph[loop_var], tf.not_equal(self.driver_param_cluster_ph[loop_var], -1))
                            typei_data = tf.gather(driver_type_param, indices=cluster_mask_filter, axis=0)
                            mui = tf.reduce_mean(typei_data, axis=0)
                            mui_list = tf.concat([mui_list, tf.expand_dims(mui, axis=0)], axis=0)
                            return [loop_var + 1, mui_list]
                        _within_condition = lambda loop_var, m: tf.less(loop_var + self.driver_cluster_days, self.n_steps)
                        def _within_body(loop_var, sigma_within_sum):
                            # note: shape of driver type param is: [days_d1, days_d2, days_d3]
                            typei_data = tf.gather(driver_type_param_seq,
                                                   indices=tf.range(loop_var, loop_var + int(self.driver_cluster_days)),
                                                   axis=1)
                            mui = tf.reduce_mean(typei_data, axis=1, keepdims=True)
                            sigma_in_i = tf.reduce_sum(tf.reduce_mean(tf.square(typei_data - mui), axis=-1))
                            sigma_within_sum += sigma_in_i
                            return [loop_var + 1, sigma_within_sum]

                        numi, sigma_within_sum = tf.while_loop(cond=_within_condition, body=_within_body, loop_vars=[i0, m0])
                        numj, mui_list = tf.while_loop(cond=_between_condition, body=_between_body, loop_vars=[i0, m_bet],
                                                       shape_invariants=[i0.get_shape(), tf.TensorShape([None, self.env_params_size])])
                        self.sigma_within_mean = sigma_within_mean = sigma_within_sum / (tf.cast(numi, tf.float32) *
                                                                tf.cast(driver_type_param_seq.shape[0], tf.float32))
                        mui_list = mui_list[1:] # zero index is a zero initialze value.
                        mu_bet_mean = tf.reduce_mean(mui_list, axis=-1, keepdims=True)
                        self.sigma_between_mean = sigma_between_mean = tf.reduce_mean(tf.square(mui_list - mu_bet_mean))
                        if self.cliped_lda:
                            clip_sigma_within_mean = tf.reduce_max([sigma_within_mean, 1e-3])
                            clip_sigma_between_mean = tf.reduce_min([sigma_between_mean, 0.3])
                            lda_loss = (clip_sigma_within_mean + 1e-6) / (clip_sigma_between_mean + 1e-6)
                        elif self.square_lda:
                            square_sigma_within_mean = tf.square(sigma_within_mean - (1e-3))#  + tf.random_uniform((), minval=0, maxval=1e-4)))
                            square_sigma_between_mean = tf.square(sigma_between_mean - (0.5))#  + tf.random_uniform((), minval=0, maxval=0.05)))
                            lda_loss = square_sigma_between_mean + square_sigma_within_mean
                        else:
                            lda_loss = (sigma_within_mean + 1e-6) / (sigma_between_mean + 1e-6)
                        if self.log_lda:
                            lda_loss = tf.log(lda_loss)
                        self.lda_loss = lda_loss
                        # bin by mean and std -- you need a env index
                        # Value function clipping: not present in the original PPO
                        if self.cliprange_vf is None:
                            # Default behavior (legacy from OpenAI baselines):
                            # use the same clipping as for the policy
                            self.clip_range_vf_ph = self.clip_range_ph
                            self.cliprange_vf = self.cliprange
                        elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                            # Original PPO implementation: no value function clipping
                            self.clip_range_vf_ph = None
                        else:
                            # Last possible behavior: clipping range
                            # specific to the value function
                            self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                        if self.clip_range_vf_ph is None:
                            # No clipping
                            vpred_clipped = train_model.value_flat
                        else:
                            # Clip the different between old and new value
                            # NOTE: this depends on the reward scaling
                            vpred_clipped = self.old_vpred_ph + \
                                            tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)

                        vf_losses1 = tf.square(vpred - self.rewards_ph)
                        vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                        self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                        ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                        ratio_fix_env = tf.exp(self.old_neglog_pac_ph - neglogpac_fix_env)
                        pg_losses = -self.advs_ph * ratio
                        pg_losses_fix_env = -self.advs_ph * ratio_fix_env
                        pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                      self.clip_range_ph)
                        pg_losses2_fix_env = -self.advs_ph * tf.clip_by_value(ratio_fix_env, 1.0 - self.clip_range_ph, 1.0 +
                                                                      self.clip_range_ph)
                        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                        self.pg_loss_fix_env = tf.reduce_mean(tf.maximum(pg_losses_fix_env, pg_losses2_fix_env))
                        self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                          self.clip_range_ph), tf.float32))

                        if self.l2_reg_pi > 0:
                            l2_vars = tf_util.get_trainable_vars(self.name + '/' + self.act_model.name + '/pi')
                            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_pi
                            self.pg_loss += l2_loss
                            self.pg_loss_fix_env += l2_loss
                        if self.l2_reg_v > 0:
                            l2_vars = tf_util.get_trainable_vars(self.name + '/' + self.act_model.name + '/vf')
                            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_v
                            self.vf_loss += l2_loss

                        if self.l2_reg_env > 0:
                            l2_vars = tf_util.get_trainable_vars(self.name + '/' + self.env_extractor_act_model.name + '/pi')
                            l2_vars += tf_util.get_trainable_vars(self.name + '/' + self.env_extractor_act_model.name + '/lstm')
                            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_env
                            self.pg_loss += l2_loss
                        pg_loss_ent = self.pg_loss - self.entropy * self.ent_coef
                        pg_loss_ent_fix_env = self.pg_loss_fix_env - self.entropy_fix_env * self.ent_coef

                        with tf.variable_scope('model'):
                            self.params = tf.trainable_variables()
                            if self.full_tensorboard_log:
                                for var in self.params:
                                    tf.summary.histogram(var.name, var)
                        p_grads = tf.gradients(pg_loss_ent, self.params)
                        p_grads_fix_env = tf.gradients(pg_loss_ent_fix_env, self.params)
                        env_extractor_var = tf.trainable_variables(self.name + '/' + env_extractor_act_model.name)
                        policy_var = tf.trainable_variables(self.name + '/' + act_model.name)
                        lda_grads = tf.gradients(lda_loss, env_extractor_var)
                        pgv_grads = tf.gradients(pg_loss_ent, env_extractor_var)
                        lda_grads_proj = []
                        grads_proj = []
                        inner_prods = []
                        env_grads_with_lda = []
                        # projection lda = error_vector = g_lda - p = g_lda - (g_pg * g_lda) / (g_pg * g_pg)
                        for lda_g, pgv_g in zip(lda_grads, pgv_grads):
                            if lda_g is None:
                                assert pgv_g is None
                                lda_grads_proj.append(None)
                                inner_prods.append(None)
                                grads_proj.append(None)
                                env_grads_with_lda.append(None)
                                continue

                            if self.constraint_lda == ConstraintLDA.NO:
                                env_grads_with_lda.append(pgv_g + lda_g)
                            elif self.constraint_lda == ConstraintLDA.Proj2PG:
                                inner_prod = tf.multiply(pgv_g, lda_g)
                                grad_proj = tf.multiply(pgv_g, inner_prod / (tf.multiply(pgv_g, pgv_g) + 1e-6))
                                lda_g_proj = lda_g - grad_proj
                                if self.scaled_lda:
                                    lda_l2_loss = tf.nn.l2_loss(lda_g_proj)
                                    policy_l2_loss = tf.nn.l2_loss(pgv_g)
                                    lda_g_proj = lda_g_proj * policy_l2_loss / tf.reduce_max([lda_l2_loss, policy_l2_loss])
                                env_grads_with_lda.append(lda_g_proj + pgv_g)
                                lda_grads_proj.append(lda_g_proj)
                                inner_prods.append(inner_prod)
                                grads_proj.append(grad_proj)
                            elif self.constraint_lda == ConstraintLDA.Proj2CGC:
                                inner_prod = tf.multiply(lda_g, pgv_g)
                                grad_proj = tf.multiply(lda_g, inner_prod / (tf.multiply(lda_g, lda_g) + 1e-6)) # CGC方向的头像
                                lda_g_proj = pgv_g - grad_proj # CGC垂直方向的PG投影
                                if self.scaled_lda:
                                    lda_l2_loss = tf.reduce_max([tf.nn.l2_loss(lda_g), 0.5])
                                    policy_l2_loss = tf.nn.l2_loss(lda_g_proj)
                                    lda_g_proj = lda_g_proj * lda_l2_loss / tf.reduce_max([lda_l2_loss, policy_l2_loss])
                                env_grads_with_lda.append(lda_g_proj + lda_g)
                                lda_grads_proj.append(lda_g_proj)
                                inner_prods.append(inner_prod)
                                grads_proj.append(grad_proj)
                            else:
                                raise NotImplementedError


                        if self.lda_max_grad_norm > 0:
                            env_grads_with_lda, _lda_grad_norm = tf.clip_by_global_norm(env_grads_with_lda,
                                                                                    self.lda_max_grad_norm)
                        if self.use_lda_loss:
                            # merge gradient
                            assign_counter = 0
                            for i in range(len(p_grads)):
                                grad = p_grads[i]
                                param = self.params[i]
                                if grad is not None:
                                    if param in env_extractor_var:
                                        print(param)
                                        idx = env_extractor_var.index(param)
                                        p_grads[i] = env_grads_with_lda[idx]
                                        assign_counter += 1
                            assert assign_counter == len([i for i in env_grads_with_lda if i is not None])
                        p_original_grads = p_grads
                        p_original_grads_fix_env = p_grads_fix_env
                        if self.p_grad_norm > 0:
                            p_grads, _p_grad_norm = tf.clip_by_global_norm(p_original_grads,
                                                                           self.p_grad_norm)
                            p_grads_fix_env, _p_grad_norm_fix_env = tf.clip_by_global_norm(p_original_grads_fix_env,
                                                                                           self.p_grad_norm)
                        p_grads = list(zip(p_grads, self.params))
                        p_grads_fix_env = list(zip(p_grads_fix_env, self.params))
                        v_original_grads = tf.gradients(self.vf_loss, self.params)
                        if self.v_grad_norm > 0:
                            v_grads, _v_grads_norm = tf.clip_by_global_norm(v_original_grads, self.v_grad_norm)
                        v_grads = list(zip(v_grads, self.params))
                    with tf.variable_scope("loss_info", reuse=False):
                        tf.summary.scalar('entropy_loss', self.entropy)
                        tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                        tf.summary.scalar('value_function_loss1', tf.reduce_mean(vf_losses1))
                        tf.summary.scalar('value_function_loss2 (clip)', tf.reduce_mean(vf_losses2))
                        tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                        tf.summary.scalar('clip_factor', self.clipfrac)
                        tf.summary.scalar('pg_loss_ent', pg_loss_ent)
                        tf.summary.scalar('_g_grad_norm', _p_grad_norm)
                        tf.summary.scalar('_g_grad_norm_fix_env', _p_grad_norm_fix_env)
                        tf.summary.scalar('_v_grads_norm', _v_grads_norm)
                        if self.lda_max_grad_norm > 0:
                            tf.summary.scalar('_lda_grad_norm', _lda_grad_norm)
                        tf.summary.scalar('lda_loss', lda_loss)
                        tf.summary.scalar('sigma_within_mean', sigma_within_mean)
                        tf.summary.scalar('sigma_between_mean', sigma_between_mean)
                    if self.rms_opt:
                        v_trainer = tf.train.RMSPropOptimizer(learning_rate=self.v_learning_rate_ph)
                        g_trainer = tf.train.RMSPropOptimizer(learning_rate=self.p_learning_rate_ph)
                        g_trainer_env = tf.train.RMSPropOptimizer(learning_rate=self.p_env_learning_rate_ph)

                    else:
                        v_trainer = tf.train.AdamOptimizer(learning_rate=self.v_learning_rate_ph, epsilon=1e-5)
                        g_trainer = tf.train.AdamOptimizer(learning_rate=self.p_learning_rate_ph, epsilon=1e-5)
                        g_trainer_env = tf.train.AdamOptimizer(learning_rate=self.p_env_learning_rate_ph, epsilon=1e-5)

                    self._v_train = v_trainer.apply_gradients(v_grads)
                    self._g_train = g_trainer.apply_gradients(p_grads)
                    def has_grad_test(vgs, name):
                        for vg in vgs:
                            if vg[0] is not None:
                                logger.info("{}: {}".format(name, vg[1]))
                    has_grad_test(v_grads, "v_grads")
                    has_grad_test(p_grads, "p_grads")
                    has_grad_test(p_grads_fix_env, "p_grads_fix_env")
                    p_grads_policy = []
                    p_grads_env = []
                    for pg in p_grads:
                        var = pg[1]
                        if var in policy_var:
                            p_grads_policy.append(pg)
                            logger.info("in policy", pg)
                        elif var in env_extractor_var:
                            p_grads_env.append(pg)
                            logger.info("in env extractor", pg)
                        else:
                            if pg[0] is not None:
                                logger.warn("unexpected var", pg)
                                raise Exception("gradient error!")
                    # training policy
                    self._g_train_policy = g_trainer.apply_gradients(p_grads_fix_env)
                    self._g_train_env = g_trainer_env.apply_gradients(p_grads)
                    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    if self.normalize_rew:
                        tf.summary.scalar('rew_mean', self.rew_rms.mean)
                        tf.summary.scalar('rew_std', self.rew_rms.std)
                    tf.summary.scalar('p_learning_rate', tf.reduce_mean(self.p_learning_rate_ph))
                    tf.summary.scalar('v_learning_rate', tf.reduce_mean(self.v_learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                    batch_env_param = tf.reshape(old_driver_type_param, [-1, self.n_steps, self.env_params_size])
                    env_param_mean, env_param_std = tf.nn.moments(batch_env_param, axes=1)
                    env_param_std = tf.reduce_mean(env_param_std)
                    env_param_mean_std = tf.nn.moments(env_param_mean, axes=0)[1]
                    env_param_mean_std = tf.reduce_mean(env_param_mean_std)
                    tf.summary.scalar('scatter_of_driver', env_param_mean_std)
                    tf.summary.scalar('scatter_of_timestep', env_param_std)

                self.summary = tf.summary.merge_all()
                self.fos_ph = tf.placeholder(tf.float32, [None], name="fos_ph")
                self.cost_ph = tf.placeholder(tf.float32, [None], name="cost_ph")
                self.coupon_ph = tf.placeholder(tf.float32, [None, 5], name="coupon_ph")
                self.domain_summary_dict = {}
                self.rew_ph = tf.placeholder(tf.float32, [None], name="fos_ph")
                self.acs_ph = tf.placeholder(tf.float32, [None], name="acs_ph")
                self.satisfication_ph = tf.placeholder(tf.float32, [None], name="satisficiation_ph")
                self.domain_summary_dict = {}
                for domain in self.all_domain_list:
                    with tf.variable_scope("{}-distribution_info".format(domain), reuse=False):
                        domain_distribution_summary = []
                        domain_distribution_summary.append(tf.summary.histogram('{}-rew'.format(domain), self.rew_ph))
                        domain_distribution_summary.append(tf.summary.histogram('{}-acs'.format(domain), self.acs_ph))
                        domain_distribution_summary.append(tf.summary.histogram('{}-satisficiation'.format(domain), self.satisfication_ph))
                        self.domain_summary_dict[domain] = tf.summary.merge(domain_distribution_summary)

                gradient_summary = []

                gradient_summary.append(tf.summary.histogram('hidden_state', old_driver_type_param))
                if self.full_tensorboard_log and self.record_gradient:
                    for i in range(len(p_grads)):
                        p_grad, param = p_grads[i]
                        p_original_grad = p_original_grads[i]
                        v_grad, param = v_grads[i]
                        v_original_grad = v_original_grads[i]
                        if p_grad is not None:
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/p_clip_grad', p_grad))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/p_grad', p_original_grad))
                        if v_grad is not None:
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/v_clip_grad', v_grad))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/v_grad', v_original_grad))
                        gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/weight', param))
                if self.constraint_lda != ConstraintLDA.NO:
                    for i in range(len(lda_grads_proj)):
                        lda_grad_proj = lda_grads_proj[i]
                        env_grad_with_lda = env_grads_with_lda[i]
                        pgv_grad = pgv_grads[i]
                        lda_grad = lda_grads[i]
                        grad_proj = grads_proj[i]
                        inner_prod = inner_prods[i]
                        param = env_extractor_var[i]
                        if lda_grad_proj is not None:
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/lda_grad_proj', lda_grad_proj))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/env_grad_with_lda', env_grad_with_lda))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/pgv_grad', pgv_grad))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/lda_grad', lda_grad))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/grads_proj', grad_proj))
                            gradient_summary.append(tf.summary.histogram(str(param.name).replace('/', '-') + '/inner_prods', inner_prod))
                self.gradient_summary = tf.summary.merge(gradient_summary)


    def record_distribution(self, acs, rews, satisfication, domain):
        distri_summary = self.sess.run(self.domain_summary_dict[domain], feed_dict={
            self.acs_ph: acs.reshape(-1, ),
            self.rew_ph: rews.reshape(-1, ),
            self.satisfication_ph: satisfication.reshape(-1, ),
        })
        tester.add_summary(distri_summary, domain + '-' + 'dis')


    def step_env_param(self, pre_obs, pre_acs, obs, state, old_state, done, domain=None):
        pre_obs_acs = np.concatenate([pre_obs, pre_acs, obs], axis=-1)
        z_t = self.step_state_distribution(pre_obs, pre_acs, obs, done, domain)
        pre_obs_acs_obs = pre_obs_acs
        if domain is None:
            psi_t, hid_state = self.env_extractor_act_model.step(pre_obs_acs_obs, state, dis_emb=z_t, mask=done)
            old_psi_t, old_hid_state = self.old_env_extractor_act_model.step(pre_obs_acs_obs, old_state, dis_emb=z_t, mask=done)
        else:
            psi_t, hid_state = self.env_extractor_eval_dict[domain].step(pre_obs_acs_obs, state, dis_emb=z_t, mask=done)
            old_psi_t, old_hid_state = self.old_env_extractor_eval_dict[domain].step(pre_obs_acs_obs, old_state, dis_emb=z_t, mask=done)
        return psi_t, old_psi_t, hid_state, old_hid_state

    def step_state_distribution(self, pre_obs, pre_acs, obs, done, domain):
        pre_obs_acs = np.concatenate([pre_obs, pre_acs, obs], axis=-1)
        if self.use_distribution_embedding:
            assert self.stable_dist_embd
            z_t = self.env.selected_env.env_z_info
            # filter_pre_obs_acs = self.distribution_embedding.filter_norm_obs(pre_obs_acs, self.env.feature_name)
            # if self.hand_code_distribution:
            #     z_t = self.sess.run(self.hand_code_distribution_info, feed_dict={self.daily_data_distribution_ph: filter_pre_obs_acs})
            # else:
            #     z_t = self.distribution_embedding.z_step(filter_pre_obs_acs)
        else:
            z_t = self.default_distribution_embedding
        z_t = np.repeat(np.expand_dims(z_t, axis=0), repeats=pre_obs_acs.shape[0], axis=0)
        return z_t

    def step(self, pre_obs, pre_acs, obs, state, old_state, policy_states, done, domain=None, deterministic=False):
        psi_t, old_psi_t, hid_state, old_hid_state = self.step_env_param(pre_obs, pre_acs, obs, state, old_state, done, domain)
        z_t = self.step_state_distribution(pre_obs, pre_acs, obs, done, domain)
        if domain is None:
            actions, values, next_policy_states, neglogpacs = self.act_model.step(obs, env_param=psi_t,old_env_param=old_psi_t,
                                                                 state_distribution_emb=z_t,
                                                                 mask=done, deterministic=deterministic,
                                                                           policy_states=policy_states)
            policy_mean, policy_std = self.act_model.proba_step(obs, env_param=psi_t,old_env_param=old_psi_t,
                                                                 state_distribution_emb=z_t,
                                                                 mask=done, policy_states=policy_states)
        else:
            actions, values, next_policy_states, neglogpacs = self.policy_eval_dict[domain].step(obs, env_param=psi_t, old_env_param=old_psi_t,
                                                                              state_distribution_emb=z_t, mask=done,
                                                                              deterministic=deterministic,
                                                                              policy_states=policy_states)
            policy_mean, policy_std = self.policy_eval_dict[domain].proba_step(obs, env_param=psi_t, old_env_param=old_psi_t,
                                                                state_distribution_emb=z_t,
                                                                mask=done,
                                                                policy_states=policy_states)

        return actions, values, hid_state, old_hid_state, next_policy_states, neglogpacs, z_t, policy_mean, policy_std


    def value(self, pre_obs, pre_acs, obs, state, old_state, policy_states, done, domain=None):
        psi_t, old_psi_t, hid_state, old_hid_state = self.step_env_param(pre_obs, pre_acs,  obs, state, old_state, done, domain)
        z_t = self.step_state_distribution(pre_obs, pre_acs, obs, done, domain)
        if domain is None:
            values = self.act_model.value(obs, env_param=psi_t, old_env_param=old_psi_t, state_distribution_emb=z_t, mask=done,
                                          policy_states=policy_states)
        else:
            values = self.policy_eval_dict[domain].value(obs, env_param=psi_t, old_env_param=old_psi_t, state_distribution_emb=z_t, mask=done,
                                                       policy_states=policy_states)

        return values

    def _train_step(self, v_learning_rate, p_learning_rate, p_env_learning_rate, cliprange,
                    obs, returns, masks, actions, values, neglogpacs, pre_obs, pre_acs, z_t, driver_param_cluster, update, training_type,
                    writer, states=None, old_states=None, policy_states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        if training_type == self.TRAINING_ALL:
            policy_train_op = self._g_train
        elif training_type == self.TRAINING_POLICY:
            policy_train_op = self._g_train_policy
        elif training_type == self.TRAINING_LSTM:
            policy_train_op = self._g_train_env
        else:
            raise NotImplementedError
        pre_obs_acs = np.concatenate([pre_obs, pre_acs, obs], axis=-1)

        td_map = {
                  self.env_extractor_train_model.obs_ph: pre_obs_acs,
                  self.env_extractor_train_model.states_ph: states,
                  self.env_extractor_train_model.dones_ph: masks,
                    self.old_env_extractor_train_model.obs_ph: pre_obs_acs,
                    self.old_env_extractor_train_model.states_ph: old_states,
                    self.old_env_extractor_train_model.dones_ph: masks,
                  self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.train_model.states_ph: policy_states,
                  self.train_model.dones_ph: masks,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.v_learning_rate_ph: v_learning_rate, self.p_learning_rate_ph: p_learning_rate,
                    self.p_env_learning_rate_ph: p_env_learning_rate,
                  self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
                  self.driver_param_cluster_ph: driver_param_cluster.astype(np.int32),
                  self.distribution_emb_info: z_t
        }
        # filter_pre_obs_acs = self.distribution_embedding.filter_norm_obs(pre_obs_acs, self.env.feature_name)
        # td_map[self.daily_data_distribution_ph] = filter_pre_obs_acs
        # if states is not None:
        #     td_map[self.train_model.states_ph] = states
        #     td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % (self.log_interval * 2) == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, driver_type_param, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac,
                     self._v_train, self.driver_type_param, policy_train_op],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                if (update % (self.log_interval * 12) == 0 or update == 1):
                    summary, gradient_summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, driver_type_param, _, _ = self.sess.run(
                        [self.summary, self.gradient_summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self.driver_type_param,
                         self._v_train,  policy_train_op],
                        td_map)
                    tester.add_summary(gradient_summary, 'gradient-' + self.env.domain)
                else:
                    summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, driver_type_param, lda_loss, sigma_within_mean, sigma_between_mean, _, _ = self.sess.run(
                        [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac,
                         self.driver_type_param, self.lda_loss, self.sigma_within_mean, self.sigma_between_mean,
                         self._v_train, policy_train_op],
                        td_map)
                    if self.use_lda_loss:
                        def lda_test():
                            mui_list = []
                            for i in range(driver_param_cluster.shape[0]):
                                cluster_mask_filter = driver_param_cluster[i][np.where(driver_param_cluster[i] != -1)]
                                typei_data = driver_type_param[cluster_mask_filter]
                                mui = np.mean(typei_data, axis=0)
                                mui_list.append(mui)
                            mui_list = np.array(mui_list)
                            mu_bet_mean = np.expand_dims(np.mean(mui_list, axis=-1), axis=-1)
                            np_sigma_between_mean = np.mean(np.square(mui_list - mu_bet_mean))
                            sigma_sum = 0
                            driver_type_param_seq = driver_type_param.reshape(-1, self.n_steps, driver_type_param.shape[-1])
                            for i in range(self.n_steps - self.driver_cluster_days):
                                typei_data = driver_type_param_seq[:, i:i + self.driver_cluster_days]
                                mui = np.expand_dims(np.mean(typei_data, axis=1), axis=1)
                                sigma_i = np.sum(np.mean(np.square(typei_data - mui), axis=-1))
                                sigma_sum += sigma_i
                            np_sigma_with_mean = sigma_sum / ((self.n_steps - self.driver_cluster_days) * (driver_type_param_seq.shape[0]))
                            assert np.allclose(sigma_within_mean, np_sigma_with_mean) and np.allclose(np_sigma_between_mean, sigma_between_mean)

                        emb_unit_test_cont.add_test('lda_test',
                                                    lda_test, emb_unit_test_cont.test_type.SINGLE_TIME)
                        emb_unit_test_cont.do_test('lda_test')

            if (1 + update) % (self.log_interval) == 0:
                tester.add_summary(summary, 'input_info-' + self.env.domain, simple_val=True, freq=0)

        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _= self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac,
                 self._v_train, policy_train_op], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2", reset_num_timesteps=True):
        logger.info(" --- start learning --- ")
        # Transform to callable if needed
        self.v_learning_rate = get_schedule_fn(self.v_learning_rate)
        self.p_learning_rate = get_schedule_fn(self.p_learning_rate)
        self.p_env_learning_rate = get_schedule_fn(self.p_env_learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose):
            self._setup_learn()
            self.sess.run(self.init_updates)
            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                            train_steps=self.n_steps, horizon=self.env.time_budget, log_freq_base=log_interval, rew_normalize=self.normalize_rew)
            eval_runner = Runner(env=self.eval_env, model=self, n_steps=self.env.time_budget, gamma=self.gamma, lam=self.lam,
                            train_steps=self.n_steps, horizon=self.env.time_budget, log_freq_base=log_interval, rew_normalize=self.normalize_rew)
            self.episode_reward = np.zeros((self.n_envs,))
            sliding_len = 50
            # ep_info_buf = deque(maxlen=sliding_len * self.env.city_len)
            t_first_start = time.time()

            n_updates = int(total_timesteps // self.n_batch)
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0 #  self.n_batch = int(self.n_envs * self.n_steps)
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                v_lr_now = self.v_learning_rate(1.0 - frac)
                p_lr_now = self.p_learning_rate(1.0 - frac)
                p_env_lr_now = self.p_env_learning_rate(1.0 - frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                t_s = time.time()
                obs, returns, masks, actions, values, neglogpacs, states, old_states, policy_states, ep_infos, true_reward, \
                pre_obs, pre_actions, driver_param_clusters, z_t = runner.run(epoch=update, deterministic=False)
                t_e = time.time()
                logger.record_tabular("time_used/sample", t_e - t_s)
                self.num_timesteps += self.n_batch
                tester.time_step_holder.set_time(self.num_timesteps)
                logger.info("timestep: {} - update {}".format(self.num_timesteps, update))
                # ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                t_s = time.time()
                if states is None:  # nonrecurrent version
                    assert False, "just for env aware policy."
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(v_lr_now, p_lr_now, p_env_lr_now, cliprange_now, *slices,
                                                                 writer=tester.writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs * self.n_train_domains)
                    flat_indices = np.arange(self.n_envs * self.n_steps * self.n_train_domains).reshape(self.n_envs * self.n_train_domains, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps # 一个batch 的环境(司机)个数， self.nminibatches == 1 时，该值为环境个数
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs * self.n_train_domains, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * (self.n_envs * self.n_train_domains) + epoch_num *
                                                                            (self.n_envs * self.n_train_domains) + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel() # 每一个环境要选择全部的steps，这里是去筛选对应的steps 下标
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs, pre_obs, pre_actions, z_t))
                            mb_states = states[mb_env_inds]
                            mb_old_states = old_states[mb_env_inds]
                            mb_policy_states = policy_states[mb_env_inds]
                            assert mb_flat_inds.max() == driver_param_clusters.max()
                            t_s1 = time.time()
                            if self.lstm_train_freq == -1:
                                training_type = self.TRAINING_ALL
                            elif update % self.lstm_train_freq == 0:
                                training_type = self.TRAINING_LSTM
                            else:
                                training_type = self.TRAINING_POLICY
                            mb_loss_vals.append(self._train_step(v_lr_now, p_lr_now, p_env_lr_now, cliprange_now, *slices,
                                                                 driver_param_cluster=driver_param_clusters, update=update,
                                                                 writer=tester.writer, states=mb_states, old_states=mb_old_states,
                                                                 policy_states=mb_policy_states,
                                                                 cliprange_vf=cliprange_vf_now, training_type=training_type))
                            t_e1 = time.time()
                            logger.record_tabular("time_used/train_epoch", t_e1 - t_s1)
                if update % self.soft_update_freq == 0:
                    self.sess.run(self.soft_updates)
                t_e = time.time()
                logger.record_tabular("time_used/train", t_e - t_s)
                t_s = time.time()
                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fph = self.n_batch / ((t_now - t_start)/60/60)
                logger.record_tabular("time_used/epoch", t_now - t_start)
                if self.verbose >= 1 and (update % (log_interval) == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("info/serial_timesteps", update * self.n_steps)
                    logger.logkv("info/n_updates", update)
                    logger.logkv("info/total_timesteps", self.num_timesteps)
                    logger.logkv("info/fph", fph)
                    logger.logkv("explained_variance/{}".format(runner.env.domain), float(explained_var))
                    logger.logkv('info/time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv("loss-{}/{}".format(self.env.domain, loss_name), loss_val)
                    logger.dumpkvs()
                    t_e = time.time()
                    logger.record_tabular("time_used/epi_info_log", t_e - t_s)

                # evaluation
                if self.verbose >= 1 and (update % (log_interval * 4) == 0 or update == 1):
                    t_s = time.time()
                    def eval_func(test_env, test_runner, tag, save_data=False):
                        logger.info("evaluation {}".format(tag))
                        eval_ep_info_buf = []
                        for domain in test_env.domain_list:
                            logger.info("evaluating {}-{} ..".format(tag, domain))
                            if tag == 'test':
                                test_runner.run(evaluation=EvaluationType.EVALUATION, domain=domain, epoch=update, deterministic=False)
                            test_runner.run(evaluation=EvaluationType.EVALUATION, domain=domain, epoch=update, deterministic=True)
                    eval_func(self.eval_env, eval_runner, 'test')
                    if update % (log_interval * 8) == 0 or update == 1:
                        eval_func(self.env, runner, 'train')
                    if update % (log_interval * 12) == 0 or update == 1:
                        tester.save_checkpoint(self.num_timesteps)
                    t_e = time.time()
                    logger.record_tabular("time_used/evaluation", t_e - t_s)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
