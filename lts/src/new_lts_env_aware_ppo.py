import time
import sys
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from common.config import *
from common import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
#from lts.src.policies import RecurrentActorCriticPolicy, LstmPolicy
from transfer_learning.src.policies import ActorCriticPolicy,RecurrentActorCriticPolicy, LstmPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from common.tester import tester
from common.utils import *
from common.config import *
from common.mpi_running_mean_std import RunningMeanStd
from common.tf_func import *
from lts.src.experiment_env import SimpleEnvAwareRunner as Runner
from vae_lts.src.vae_handler import VAE


class PPO2(ActorCriticRLModel):
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

    def __init__(self, sess, policy, env_extractor_policy, vae_handler,
                 hand_code_distribution, env, eval_env, all_domain_list, random_range=0.1,
                 env_params_size=20, gamma=0.99, n_steps=128, ent_coef=0.01,
                 v_learning_rate=2.5e-4, p_learning_rate=2.5e-4, p_env_learning_rate=2.5e-4, vf_coef=0.5,
                 p_grad_norm=0.5, v_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2,
                 cliprange_vf=None, driver_cluster_days=7,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 env_extractor_policy_kwargs=None,
                 stable_dist_embd=False, l2_reg_pi=0, l2_reg_env=0, l2_reg_v=0, tau=0.001,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, name='ppo', use_lda_loss=False,
                 pi_lda_loss=False,
                 log_lda=False, merge_samples =True,
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
        self.random_range = random_range
        self.merge_samples = merge_samples

        self.cliped_lda = cliped_lda
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
        self.random_range = random_range
        self.cliprange = cliprange
        self.eval_env = eval_env
        self.log_lda = log_lda
        self.env_extractor_policy = env_extractor_policy
        self.hand_code_distribution = hand_code_distribution
        self.vae_handler = vae_handler
        self.use_lda_loss = use_lda_loss
        self.pi_lda_loss = pi_lda_loss
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

    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            if self.merge_samples:
                self.n_train_domains = len(self.env.domain_list)
            else:
                self.n_train_domains = 1
            self.n_batch = int(self.n_envs * self.n_steps * self.n_train_domains)

            self.graph = tf.compat.v1.get_default_graph()
            self.observation_space = box_concate(self.observation_space, self.action_space)
            if self.vae_handler is not None:
                assert isinstance(self.vae_handler, VAE)
                self.vae_emb_size = self.vae_handler.z_size
            else:
                self.vae_emb_size = 1
            identity = np.ones(shape=(self.vae_emb_size,))
            vae_emb_space = gym.spaces.Box(low=identity * -10, high=identity * 10, dtype=np.float32, )
            self.observation_space = box_concate(self.observation_space, vae_emb_space)
            with self.graph.as_default():
                with tf.compat.v1.variable_scope(self.name):
                    if self.normalize_rew:
                        with tf.compat.v1.variable_scope('rew_rms'):
                            self.rew_rms = RunningMeanStd()
                    else:
                        self.rew_rms = None

                    self.set_random_seed(self.seed)
                    # self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                    n_batch_step = None
                    n_batch_train = None
                    # TODO: find this policy
                    #assert issubclass(self.policy, LstmPolicy)
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                    self.act_model = act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                            n_batch_step, reuse=False, **self.policy_kwargs)
                    self.eval_model = {}
                    for domain in self.all_domain_list:
                        n_batch_step = driver_number = self.env.env_dict[domain].num_envs
                        eval_model = self.policy(self.sess, self.observation_space, self.action_space, driver_number, 1,
                                                n_batch_step, reuse=True, **self.policy_kwargs)
                        self.eval_model[domain] = eval_model
                    with tf.compat.v1.variable_scope("train_model", reuse=True,
                                           custom_getter=tf_util.outer_scope_getter("train_model")):
                        self.train_model = train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                  self.n_envs * self.n_train_domains // self.nminibatches, self.n_steps, n_batch_train,
                                                  reuse=True, **self.policy_kwargs)
                    with tf.compat.v1.variable_scope("loss", reuse=False):
                        self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                        self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                        self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")
                        self.old_neglog_pac_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                        self.old_vpred_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_vpred_ph")
                        self.v_learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="v_learning_rate_ph")
                        self.p_learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="p_learning_rate_ph")
                        self.clip_range_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_ph")

                        neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                        self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                        vpred = train_model.value_flat

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
                            self.clip_range_vf_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_vf_ph")

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
                        pg_losses = -self.advs_ph * ratio
                        pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                      self.clip_range_ph)
                        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                        self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                          self.clip_range_ph), tf.float32))


                        if self.l2_reg_pi > 0:
                            # TODO pi or v and pi
                            l2_vars = tf_util.get_trainable_vars(self.name + '/' + self.act_model.name + '/model/pi_')
                            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_pi
                            self.pg_loss += l2_loss
                        if self.l2_reg_v > 0:
                            # TODO pi or v and pi
                            l2_vars = tf_util.get_trainable_vars(self.name + '/' + self.act_model.name + '/model/vf')
                            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_v
                            self.vf_loss += l2_loss

                        pg_loss_ent = self.pg_loss - self.entropy * self.ent_coef

                        # CGC constraint
                        self.driver_param_cluster_ph = tf.compat.v1.placeholder(tf.int32, [None, None], name="driver_param_cluster_ph")

                        def compute_sigma(driver_type_param):
                            i0 = tf.constant(0)
                            m0 = tf.constant(0.0)
                            m_bet = tf.zeros(shape=(1, self.env_params_size))
                            _between_condition = lambda loop_var, m: tf.less(loop_var, tf.shape(self.driver_param_cluster_ph)[0])
                            mu = tf.reduce_mean(driver_type_param, axis=0)
                            driver_type_param_seq = tf.reshape(driver_type_param, shape=(-1, self.n_steps, driver_type_param.shape[-1]))
                            # every time-steps has a type cluster.
                            def _between_body(loop_var, mui_list):
                                cluster_mask_filter = tf.boolean_mask(self.driver_param_cluster_ph[loop_var],
                                                                      tf.not_equal(self.driver_param_cluster_ph[loop_var], -1))
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
                            sigma_within_mean = sigma_within_sum / (tf.cast(numi, tf.float32) *
                                                                    tf.cast(driver_type_param_seq.shape[0], tf.float32))
                            mui_list = mui_list[1:] # zero index is a zero initialze value.
                            mu_bet_mean = tf.reduce_mean(mui_list, axis=-1, keepdims=True)
                            sigma_between_mean = tf.reduce_mean(tf.square(mui_list - mu_bet_mean))
                            return sigma_within_mean, sigma_between_mean

                        v_driver_type_param = self.train_model.v_rnn_output
                        p_driver_type_param = self.train_model.pi_rnn_output
                        self.v_sigma_within_mean, self.v_sigma_between_mean = compute_sigma(v_driver_type_param)
                        self.pi_sigma_within_mean, self.pi_sigma_between_mean = compute_sigma(p_driver_type_param)

                        def compute_cgc_loss(sigma_within, sigma_between):
                            square_sigma_within_mean = tf.square(
                                sigma_within - (4e-3 + tf.random_uniform((), minval=0, maxval=4e-3 * self.random_range)))
                            square_sigma_between_mean = tf.square(
                                sigma_between - (0.25 + tf.random_uniform((), minval=0, maxval=0.25 * self.random_range)))
                            lda_loss = square_sigma_between_mean + square_sigma_within_mean
                            return lda_loss
                        self.pi_cgc_loss = compute_cgc_loss(self.pi_sigma_within_mean, self.pi_sigma_between_mean)
                        self.v_cgc_loss = compute_cgc_loss(self.v_sigma_within_mean, self.v_sigma_between_mean)

                        v_env_extractor_var = tf.compat.v1.trainable_variables(
                            self.name + '/' + act_model.name + '/model/lstm_v') + tf.compat.v1.trainable_variables(
                            self.name + '/' + act_model.name + '/model/vf_fc')
                        pi_env_extractor_var = tf.compat.v1.trainable_variables(
                            self.name + '/' + act_model.name + '/model/lstm_pi') + tf.compat.v1.trainable_variables(
                            self.name + '/' + act_model.name + '/model/pi_fc')

                        def cgc_projection(input_loss, lda_loss, vars):
                            lda_grads = tf.gradients(lda_loss, vars)
                            v_grads = tf.gradients(input_loss, vars)
                            lda_grads_proj = []
                            grads_proj = []
                            inner_prods = []
                            env_grads_with_lda = []
                            for lda_g, v_g in zip(lda_grads, v_grads):
                                if lda_g is None:
                                    print(v_g)
                                    assert v_g is None
                                    lda_grads_proj.append(None)
                                    inner_prods.append(None)
                                    grads_proj.append(None)
                                    env_grads_with_lda.append(None)
                                    continue

                                if self.constraint_lda == ConstraintLDA.NO:
                                    env_grads_with_lda.append(v_g + lda_g)
                                elif self.constraint_lda == ConstraintLDA.Proj2CGC:
                                    inner_prod = tf.multiply(lda_g, v_g)
                                    grad_proj = tf.multiply(lda_g,
                                                            inner_prod / (tf.multiply(lda_g, lda_g) + 1e-6))  # CGC方向的头像
                                    lda_g_proj = v_g - grad_proj  # CGC垂直方向的PG投影
                                    if self.scaled_lda:
                                        lda_l2_loss = tf.nn.l2_loss(lda_g)
                                        value_l2_loss = tf.nn.l2_loss(lda_g_proj)
                                        lda_g_proj = lda_g_proj * lda_l2_loss / tf.reduce_max([lda_l2_loss, value_l2_loss])
                                    env_grads_with_lda.append(lda_g_proj + lda_g)
                                    lda_grads_proj.append(lda_g_proj)
                                    inner_prods.append(inner_prod)
                                    grads_proj.append(grad_proj)
                                else:
                                    raise NotImplementedError
                            return env_grads_with_lda
                        v_env_grads_with_cgc = cgc_projection(self.vf_loss, self.v_cgc_loss, v_env_extractor_var)
                        pi_env_grads_with_cgc = cgc_projection(pg_loss_ent, self.pi_cgc_loss, pi_env_extractor_var)

                        with tf.compat.v1.variable_scope('model'):
                            self.params = tf.compat.v1.trainable_variables()
                            if self.full_tensorboard_log:
                                for var in self.params:
                                    tf.compat.v1.summary.histogram(var.name, var)
                        pi_original_grads = tf.gradients(pg_loss_ent, self.params)

                        v_original_grads = tf.gradients(self.vf_loss, self.params)
                        def apply_cgc(vars, original_grads, cgc_grads):
                            assign_counter = 0
                            print("cgc gradient replacement: ")
                            for i in range(len(original_grads)):
                                grad = original_grads[i]
                                param = self.params[i]
                                if grad is not None:
                                    if param in vars:
                                        print(param)
                                        idx = vars.index(param)
                                        original_grads[i] = cgc_grads[idx]
                                        assign_counter += 1
                            return original_grads

                        if self.use_lda_loss:
                            v_original_grads = apply_cgc(v_env_extractor_var, v_original_grads, v_env_grads_with_cgc)
                            if self.pi_lda_loss:
                                pi_original_grads = apply_cgc(pi_env_extractor_var, pi_original_grads, pi_env_grads_with_cgc)

                        if self.p_grad_norm > 0:
                            p_grads, _p_grad_norm = tf.clip_by_global_norm(pi_original_grads, self.p_grad_norm)
                        else:
                            p_grads = pi_original_grads
                            _p_grad_norm = 0
                        p_grads = list(zip(p_grads, self.params))
                        if self.v_grad_norm > 0:
                            v_grads, _v_grads_norm = tf.clip_by_global_norm(v_original_grads, self.v_grad_norm)
                        else:
                            v_grads = v_original_grads
                            _v_grads_norm = 0
                        v_grads = list(zip(v_grads, self.params))

                        # lda_grads = tf.gradients(lda_loss, env_extractor_var)
                    with tf.compat.v1.variable_scope("loss_info", reuse=False):
                        tf.summary.scalar('entropy_loss', self.entropy)
                        tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                        tf.summary.scalar('value_function_loss1', tf.reduce_mean(vf_losses1))
                        tf.summary.scalar('value_function_loss2 (clip)', tf.reduce_mean(vf_losses2))
                        tf.summary.scalar('value_function_loss_min', self.vf_loss)
                        tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                        tf.summary.scalar('clip_factor', self.clipfrac)
                        tf.summary.scalar('pg_loss_ent', pg_loss_ent)
                        tf.summary.scalar('_g_grad_norm', _p_grad_norm)
                        tf.summary.scalar('_v_grads_norm', _v_grads_norm)

                    v_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.p_learning_rate_ph, epsilon=1e-5)
                    g_trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_learning_rate_ph, epsilon=1e-5)
                    self._g_train = g_trainer.apply_gradients(p_grads)
                    self._v_train = v_trainer.apply_gradients(v_grads)
                    # with tf.control_dependencies([self._g_train]):
                    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    if self.normalize_rew:
                        tf.summary.scalar('rew_mean', self.rew_rms.mean)
                        tf.summary.scalar('rew_std', self.rew_rms.std)
                    tf.summary.scalar('p_learning_rate', tf.reduce_mean(self.p_learning_rate_ph))
                    tf.summary.scalar('v_learning_rate', tf.reduce_mean(self.v_learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))


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
                def compute_scatter(driver_type_param):
                    batch_env_param = tf.reshape(driver_type_param, [-1, self.n_steps, self.env_params_size])
                    env_param_mean, env_param_std = tf.nn.moments(batch_env_param, axes=1)
                    scatter_of_timestep = tf.reduce_mean(env_param_std)
                    env_param_mean_std = tf.nn.moments(env_param_mean, axes=0)[1]
                    scatter_of_driver = tf.reduce_mean(env_param_mean_std)
                    return scatter_of_driver, scatter_of_timestep
                with tf.compat.v1.variable_scope("hidden_var", reuse=False):
                    tf.summary.scalar('v_sigma_within_mean', self.v_sigma_within_mean)
                    tf.summary.scalar('v_sigma_between_mean', self.v_sigma_between_mean)
                    # sigma_within_mean, sigma_between_mean = compute_sigma(self.train_model.pi_rnn_output)
                    tf.summary.scalar('pi_sigma_within_mean', self.pi_sigma_within_mean)
                    tf.summary.scalar('pi_sigma_between_mean', self.pi_sigma_between_mean)
                    # variance of driver and timestep.
                    scatter_of_driver, scatter_of_timestep = compute_scatter(self.train_model.v_rnn_output)
                    tf.summary.scalar('v_scatter_of_driver', scatter_of_driver)
                    tf.summary.scalar('v_scatter_of_timestep', scatter_of_timestep)
                    scatter_of_driver, scatter_of_timestep = compute_scatter(self.train_model.pi_rnn_output)
                    tf.summary.scalar('pi_scatter_of_driver', scatter_of_driver)
                    tf.summary.scalar('pi_scatter_of_timestep', scatter_of_timestep)
                self.summary = tf.compat.v1.summary.merge_all()
                gradient_summary = []
                gradient_summary.append(tf.summary.histogram('v_hidden_state', v_driver_type_param))
                gradient_summary.append(tf.summary.histogram('p_hidden_state', p_driver_type_param))
                for i in range(len(p_grads)):
                    p_grad, param = p_grads[i]
                    p_original_grad = pi_original_grads[i]
                    v_grad, param = v_grads[i]
                    v_original_grad = v_original_grads[i]
                    if p_grad is not None:
                        gradient_summary.append(tf.compat.v1.summary.histogram(str(param.name).replace('/', '-') + '/p_clip_grad', p_grad))
                        gradient_summary.append(tf.compat.v1.summary.histogram(str(param.name).replace('/', '-') + '/p_grad', p_original_grad))
                    if v_grad is not None:
                        gradient_summary.append(tf.compat.v1.summary.histogram(str(param.name).replace('/', '-') + '/v_clip_grad', v_grad))
                        gradient_summary.append(tf.compat.v1.summary.histogram(str(param.name).replace('/', '-') + '/v_grad', v_original_grad))
                    gradient_summary.append(tf.compat.v1.summary.histogram(str(param.name).replace('/', '-') + '/var', param))
                self.gradient_summary = tf.compat.v1.summary.merge(gradient_summary)

                self.rew_ph = tf.compat.v1.placeholder(tf.float32, [None], name="fos_ph")
                self.acs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="acs_ph")
                self.satisfication_ph = tf.compat.v1.placeholder(tf.float32, [None], name="satisficiation_ph")
                self.domain_summary_dict = {}
                for domain in self.all_domain_list:
                    with tf.compat.v1.variable_scope("{}-distribution_info".format(domain), reuse=False):
                        domain_distribution_summary = []
                        domain_distribution_summary.append(tf.compat.v1.summary.histogram('{}-rew'.format(domain), self.rew_ph))
                        domain_distribution_summary.append(tf.compat.v1.summary.histogram('{}-acs'.format(domain), self.acs_ph))
                        domain_distribution_summary.append(tf.compat.v1.summary.histogram('{}-satisficiation'.format(domain), self.satisfication_ph))
                        self.domain_summary_dict[domain] = tf.compat.v1.summary.merge(domain_distribution_summary)


                self.proba_step = act_model.proba_step
                self.initial_state = act_model.initial_state


    def value(self, obs, vae_emb, pre_acs, pi_state, v_state, done, domain=None):
        pre_obs_acs = np.concatenate([pre_acs, vae_emb, obs], axis=-1)
        if domain is None:
            values = self.act_model.value(pre_obs_acs, pi_state=pi_state, v_state=v_state, mask=done)
        else:
            values = self.eval_model[domain].value(pre_obs_acs, pi_state=pi_state, v_state=v_state, mask=done)
        return values


    def step(self, obs, vae_emb, pre_acs, pi_state, v_state, done, deterministic=False, domain=None):
        pre_obs_acs = np.concatenate([pre_acs, vae_emb, obs], axis=-1)
        # infer vae embedding.
        if domain is None:
            actions, values, pi_state, v_state, neglogpacs = self.act_model.step(pre_obs_acs, mask=done, pi_state=pi_state, v_state=v_state, deterministic=deterministic)
            policy_mean, policy_std = self.act_model.proba_step(pre_obs_acs, pi_state=pi_state, v_state=v_state, mask=done)
        else:
            actions, values, pi_state, v_state, neglogpacs = self.eval_model[domain].step(pre_obs_acs, mask=done, pi_state=pi_state, v_state=v_state, deterministic=deterministic)
            policy_mean, policy_std = self.eval_model[domain].proba_step(pre_obs_acs, pi_state=pi_state, v_state=v_state, mask=done)
        return actions, values, pi_state, v_state, neglogpacs, policy_mean, policy_std

    def _train_step(self, v_learning_rate, p_learning_rate, cliprange, obs, vae_emb, pre_acs, returns, masks, actions, values, neglogpacs, driver_param_clusters, update,
                    writer, pi_states=None, v_states=None, cliprange_vf=None):
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
        pre_obs_acs = np.concatenate([pre_acs, vae_emb, obs], axis=-1)
        # infer vae embedding.
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: pre_obs_acs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.v_learning_rate_ph: v_learning_rate, self.p_learning_rate_ph: p_learning_rate,
                  self.clip_range_ph: cliprange,
                  self.driver_param_cluster_ph: driver_param_clusters.astype(np.int32),
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if pi_states is not None:
            td_map[self.train_model.pi_states_ph] = pi_states
            td_map[self.train_model.v_states_ph] = v_states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if pi_states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % (self.log_interval * 2) == 0:
                run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac,
                     self._v_train, self._g_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                if (update % (self.log_interval * 12) == 0 or update == 1) and self.record_gradient:
                    summary, gradient_summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _ = self.sess.run(
                        [self.summary, self.gradient_summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl,
                         self.clipfrac, self._v_train, self._g_train],
                        td_map)
                    tester.add_summary(gradient_summary, 'gradient-' + self.env.domain)
                else:
                    summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ , _= self.sess.run(
                        [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._v_train, self._g_train],
                        td_map)

            if update % self.log_interval == 0:
                tester.add_summary(summary, 'input_info-' + self.env.domain, simple_val=True, freq=0)
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._v_train, self._g_train], td_map)
        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac


    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        logger.info(" --- start learning --- ")
        # Transform to callable if needed
        self.v_learning_rate = get_schedule_fn(self.v_learning_rate)
        self.p_learning_rate = get_schedule_fn(self.p_learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose):
            self._setup_learn()

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                            train_steps=self.n_steps, horizon=self.env.time_budget, log_freq_base=log_interval, rew_normalize=self.normalize_rew,
                            merge_samples=self.merge_samples, for_update_checkpoint=False)
            eval_runner = Runner(env=self.eval_env, model=self, n_steps=self.env.time_budget, gamma=self.gamma, lam=self.lam,
                            train_steps=self.n_steps, horizon=self.env.time_budget, log_freq_base=log_interval, rew_normalize=self.normalize_rew,
                            merge_samples=self.merge_samples, for_update_checkpoint=False)
            self.episode_reward = np.zeros((self.n_envs,))
            sliding_len = 50
            t_first_start = time.time()
            n_updates = int(total_timesteps // self.n_batch)
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                v_lr_now = self.v_learning_rate(1.0 - frac)
                p_lr_now = self.p_learning_rate(1.0 - frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                keep_did = False
                obs, vae_emb, pre_acs, driver_param_clusters, returns, masks, actions, values, neglogpacs, pi_states, v_states, ep_infos, true_reward = runner.run(keep_did=keep_did,
                                                                                                             epoch=update, deterministic=False)
                self.num_timesteps += self.n_batch
                tester.time_step_holder.set_time(self.num_timesteps)

                mb_loss_vals = []
                if pi_states[0] is None:  # nonrecurrent version
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
                            mb_loss_vals.append(self._train_step(v_lr_now, p_lr_now, cliprange_now, *slices, writer=tester.writer,
                                                                 update=update, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs * self.n_train_domains)
                    flat_indices = np.arange(self.n_envs * self.n_train_domains * self.n_steps).reshape(self.n_envs * self.n_train_domains , self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs * self.n_train_domains , envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs * self.n_train_domains + epoch_num *
                                                                            self.n_envs * self.n_train_domains + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, vae_emb, pre_acs, returns, masks, actions, values, neglogpacs))
                            mb_pi_states = pi_states[mb_env_inds]
                            mb_v_states = v_states[mb_env_inds]
                            assert mb_flat_inds.max() == driver_param_clusters.max()
                            mb_loss_vals.append(self._train_step(v_lr_now, p_lr_now, cliprange_now, *slices, update=update,
                                                                 driver_param_clusters=driver_param_clusters,
                                                                 writer=tester.writer, pi_states=mb_pi_states, v_states=mb_v_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fph = self.n_batch / ((t_now - t_start)/60/60)
                logger.info("timestep: {} - update {}".format(self.num_timesteps, update))
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
                # evaluation
                if self.verbose >= 1 and (update % (log_interval * 4) == 0 or update == 1):
                    def eval_func(test_env, test_runner, tag, save_data=False):
                        logger.info("evaluation {}".format(tag))
                        for domain in test_env.domain_list:
                            logger.info("evaluating {}-{} ..".format(tag, domain))
                            test_runner.run(evaluation=EvaluationType.EVALUATION, domain=domain, epoch=update, deterministic=False)
                            test_runner.run(evaluation=EvaluationType.EVALUATION, domain=domain, epoch=update, deterministic=True)

                    eval_func(self.eval_env, eval_runner, 'test')
                    if update % (log_interval * 8) == 0 or update == 1:
                        eval_func(self.env, runner, 'train')
                    # if update % (log_interval * 12) == 0 or update == 1:
                    #     tester.save_checkpoint(self.num_timesteps)
                    logger.dumpkvs()
            return self

    def record_distribution(self, acs, rews, satisfication, domain):
        distri_summary = self.sess.run(self.domain_summary_dict[domain], feed_dict={
            self.acs_ph: acs.reshape(-1, ),
            self.rew_ph: rews.reshape(-1, ),
            self.satisfication_ph: satisfication.reshape(-1, ),
        })
        tester.add_summary(distri_summary, domain + '-' + 'dis')

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
