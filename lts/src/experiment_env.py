from abc import ABC, abstractmethod
from common.config import *
import numpy as np
from common.tf_func import *
import gym
from common import logger
from common.emb_unit_test import emb_unit_test_cont
from common.tester import tester

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, n_steps):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        # n_env = env.num_envs
        # self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        # self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        # self.pre_acs = env.restore_action(np.zeros((n_env,) + env.action_space.shape))
        # self.dones = [False for _ in range(n_env)]
        # self.obs[:] = env.reset()
        # self.pre_obs = self.obs * 0
        self.n_steps = n_steps
        self.states = model.initial_state


    @abstractmethod
    def run(self, epoch):
        """
        Run a learning step of the model
        """
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, rew_normalize, n_steps, merge_samples, for_update_checkpoint, gamma=None, lam=None, horizon, train_steps,
                 log_freq_base):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.rew_normalize = rew_normalize
        self.lam = lam
        self.gamma = gamma
        self.n_steps = n_steps
        self.horizon = horizon
        self.train_steps = train_steps
        self.states, self.dones, self.obs = {}, {}, {}
        self.multi_epi_info = {}
        self.multi_ts = {}
        self.log_freq_base = log_freq_base
        self.merge_samples = merge_samples
        self.last_best_rew = -1 * np.inf
        self.for_update_checkpoint = for_update_checkpoint

    def update_base_info(self, obs, domain):
        self.n_env = self.env.current_num_envs
        n_steps = self.n_steps
        env = self.env
        self.batch_ob_shape = (self.n_env*n_steps,) + env.observation_space.shape
        print('self.n_env', self.n_env)
        self.dones[domain] = np.array([False for _ in range(self.n_env)])
        self.obs[domain] = obs
        self.states[domain] = self.model.initial_state
        self.multi_epi_info[domain] = self.init_epi_info(domain)
        self.multi_ts[domain] = 0

    def run(self, epoch, domain=None, evaluation=EvaluationType.NO,
            dynamic_type=DynamicType.POLICY, deterministic=None, one_step=False, keep_did=False,):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """

        assert self.env.need_reset, "domain {}, evaluation {}".format(domain, evaluation)
        multi_mb_obs, multi_mb_rewards, multi_mb_actions, multi_mb_values, multi_mb_dones, multi_mb_neglogpacs = [], [], [], [], [], []
        multi_mb_returns, multi_mb_states, multi_ep_infos, multi_true_reward = [], [], [], []
        self.epoch = epoch
        self.deterministic = deterministic
        if evaluation == EvaluationType.NO:
            assert domain is None
            if self.merge_samples:
                domain_list = self.env.domain_list
            else:
                rand_indx = np.random.randint(0, len(self.env.domain_list))
                domain_list = [self.env.domain_list[rand_indx]]
            self.n_steps = self.train_steps
        else:
            domain_list = [domain]
            self.n_steps = self.horizon

        for domain in domain_list:
            self.env.change_domain(domain)
            self.domain = domain
            if self.env.need_reset(domain) or evaluation != EvaluationType.NO or epoch == 1:
                obs = self.env.reset(domain=domain, evaluation=evaluation)
                self.update_base_info(obs, domain)

            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
            mb_states = self.states[domain]
            ep_infos = []
            for _ in range(self.n_steps):
                if self.env.need_reset(domain):
                    self.log_epi_info(evaluation=evaluation)
                    obs = self.env.reset(domain=domain, evaluation=evaluation)
                    self.update_base_info(obs, domain)
                print('self.dones[domain]', self.dones[domain])
                actions, values, self.states[domain], neglogpacs, policy_mean, policy_std = self.model.step(self.obs[domain],
                                                                           self.states[domain], self.dones[domain], deterministic=deterministic)
                print('np.isnan(actions)', np.isnan(actions))
                #assert np.where(np.isnan(actions))[0].shape[0] == 0
                mb_obs.append(self.obs[domain].copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                print('mb_dones', mb_dones)
                mb_dones.append(self.dones[domain])
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.env.action_space.low.T, self.env.action_space.high.T)
                self.obs[domain][:], rewards, self.dones[domain], infos = self.env.step(clipped_actions.T)
                self.append_log_info(mb_obs=self.obs[domain], mb_rewards=rewards, info=infos, acs=clipped_actions)
                maybe_ep_info = infos.get('episode')
                # if maybe_ep_info is not None:
                #     self.log_epi_info(maybe_ep_info, evaluation)
                mb_rewards.append(rewards)
                self.multi_ts[self.domain] += 1
            if self.env.need_reset(domain):
                self.log_epi_info(evaluation=evaluation)
                # obs = self.env.reset(domain=domain, evaluation=evaluation)
                # self.update_base_info(obs, domain)
            # assert self.env.need_reset, "{} -> {} {}".format(self.env.selected_env.timestep, self.env.selected_env.T, self.n_steps)
            # compute gamma ret:
            if self.rew_normalize:
                mb_gamma_ret = np.zeros(mb_rewards[0].shape)
                for step in reversed(range(self.n_steps)):
                    mb_gamma_ret = mb_gamma_ret * self.gamma + mb_rewards[step]
                # mb_gamma_ret = swap_and_flatten(mb_gamma_ret)
                self.model.rew_rms.update(mb_gamma_ret.flatten())
            # batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=self.obs[domain].dtype)

            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            print('mb_dones', mb_dones)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            last_values = self.model.value(self.obs[domain], self.states[domain], self.dones[domain])
            true_reward = np.copy(mb_rewards)
            mb_returns = self._compute_gae_ret(mb_rewards, domain, mb_dones, last_values, mb_values)
            mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
                map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))
            multi_mb_obs.append(mb_obs)
            multi_mb_returns.append(mb_returns)
            multi_mb_dones.append(mb_dones)
            multi_mb_actions.append(mb_actions)
            multi_mb_values.append(mb_values)
            if mb_states is None:
                multi_mb_states.append([mb_states])
            else:
                multi_mb_states.append(mb_states)
            multi_ep_infos.append(ep_infos)
            multi_mb_neglogpacs.append(mb_neglogpacs)
            multi_true_reward.append(true_reward)
        multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
        multi_mb_states,  multi_ep_infos, multi_mb_neglogpacs, multi_true_reward, \
        = map(self.multi_concat, (multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions,
                             multi_mb_values, multi_mb_states, multi_ep_infos, multi_mb_neglogpacs,  multi_true_reward))

        return multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
        multi_mb_neglogpacs, multi_mb_states,  multi_ep_infos, multi_true_reward

    def multi_concat(self, inp):
        return np.concatenate(inp, axis=0)

    def _compute_gae_ret(self, mb_rewards, domain, mb_dones, last_values, mb_values):
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones[domain]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            if self.rew_normalize:
                mb_rewards[step] = normalize_np(mb_rewards[step], self.model.rew_rms, self.model.sess, just_scale=True)
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values
        return mb_returns

    def init_epi_info(self, domain):
        multidomain_env = self.env.env_dict
        return {
            "daily_rews": np.zeros([self.horizon, self.n_env]),
            "daily_acs": np.zeros([self.horizon, self.n_env, multidomain_env[domain].env._environment.num_product]),
            "daily_satisfication": np.zeros([self.horizon, self.n_env]),
        }

    def append_log_info(self, mb_rewards, mb_obs, acs, info):
        print('mb_rewards.shape', mb_rewards.shape)
        print('acs.shape', acs.shape)
        self.multi_epi_info[self.domain]['daily_rews'][self.multi_ts[self.domain]] = mb_rewards
        print('acs.shape', acs.shape)
        print('self.multi_epi_info[self.domain][daily_acs][self.multi_ts[self.domain]]', self.multi_epi_info[self.domain]['daily_acs'][self.multi_ts[self.domain]])
        self.multi_epi_info[self.domain]['daily_acs'][self.multi_ts[self.domain]] = acs
        #self.multi_epi_info[self.domain]['daily_satisfication'][self.multi_ts[self.domain]] = info['hidden_state'][:, self.env.hd_feature.index('satisfication')]
        self.multi_epi_info[self.domain]['daily_satisfication'][self.multi_ts[self.domain]] = None

    def log_epi_info(self, evaluation):
        if evaluation == EvaluationType.NO:
            prefix = 'epi_info-{}/'.format(self.domain)
        else:
            if self.deterministic:
                prefix = 'evaluation-{}/'.format(self.domain)
            else:
                prefix = 'evaluation-stoc-{}/'.format(self.domain)
        def log_mean(key):
            logger.record_tabular(prefix + key, np.mean(np.mean(self.multi_epi_info[self.domain][key], axis=0), axis=0))

        def log_sum(key):
            logger.record_tabular(prefix + key, np.mean(np.sum(self.multi_epi_info[self.domain][key], axis=0), axis=0))
        if self.epoch % self.log_freq_base == 0 or self.epoch == 1:
            list(map(log_mean, ('daily_acs', 'daily_satisfication')))
            list(map(log_sum, ('daily_rews', )))
            if evaluation != EvaluationType.NO:
                if tester.hyper_param['plt_res']:
                    prefix = 'performance/{}-stoc-{}/acs-{}'.format(self.domain, self.deterministic, self.epoch)
                    acs_distribution = self.multi_epi_info[self.domain]['daily_acs'].flatten()
                    tester.simple_hist(prefix, [acs_distribution], ['Upper Bound'], pretty=True, xlabel='clickbaitiness score', ylabel='density',
                                        bins=100, density=True, log=False)

                self.model.record_distribution(acs=self.multi_epi_info[self.domain]['daily_acs'],
                                               rews=self.multi_epi_info[self.domain]['daily_rews'],
                                               satisfication=self.multi_epi_info[self.domain]['daily_satisfication'],
                                               domain=self.domain)
                perf = np.mean(np.sum(self.multi_epi_info[self.domain]['daily_rews']))
                if perf > self.last_best_rew and self.for_update_checkpoint:
                    logger.info("better performance: {} > {}".format(perf, self.last_best_rew))
                    tester.save_checkpoint(self.epoch)
                    self.last_best_rew = perf
        #logger.dump_tabular()

class SimpleEnvAwareRunner(Runner):
    def __init__(self, *args, **kwargs):
        super(SimpleEnvAwareRunner, self).__init__( *args, **kwargs)
        self.pi_states, self.v_states,  self.dones, self.obs, self.pre_acs, self.vae_emb = {}, {}, {}, {}, {}, {}

    def update_base_info(self, obs, domain):
        self.n_env = self.env.current_num_envs
        n_steps = self.n_steps
        env = self.env
        self.batch_ob_shape = (self.n_env*n_steps,) + env.observation_space.shape
        self.dones[domain] = np.array([False for _ in range(self.n_env)])
        self.obs[domain] = obs
        self.pre_acs[domain] = np.zeros((self.env.current_num_envs,) + self.env.action_space.shape)
        self.pi_states[domain] = np.zeros((self.env.current_num_envs, self.model.initial_state.shape[1]))
        self.v_states[domain] = np.zeros((self.env.current_num_envs, self.model.initial_state.shape[1]))
        self.vae_emb[domain] = np.zeros((self.env.current_num_envs, self.model.vae_emb_size))
        self.multi_epi_info[domain] = self.init_epi_info(domain)

        self.multi_ts[domain] = 0

    def run(self, epoch, domain=None, evaluation=EvaluationType.NO,
            dynamic_type=DynamicType.POLICY, deterministic=None, one_step=False, keep_did=False,):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """

        assert self.env.need_reset, "domain {}, evaluation {}".format(domain, evaluation)
        multi_mb_obs, multi_mb_vae_emb,  multi_mb_pre_actions, multi_mb_rewards, multi_mb_actions, multi_mb_values, multi_mb_dones, multi_mb_neglogpacs = [], [], [], [], [], [], [], []
        multi_mb_returns, multi_mb_pi_states,  multi_mb_v_states, multi_ep_infos, multi_true_reward = [], [], [], [], []
        multi_cluster_hash = []
        self.epoch = epoch
        self.deterministic = deterministic
        if evaluation == EvaluationType.NO:
            assert domain is None
            domain_list = self.env.domain_list
            self.n_steps = self.train_steps
            if self.merge_samples:
                domain_list = self.env.domain_list
            else:
                rand_indx = np.random.randint(0, len(self.env.domain_list))
                domain_list = [self.env.domain_list[rand_indx]]
        else:
            domain_list = [domain]
            self.n_steps = self.horizon

        for domain in domain_list:
            self.env.change_domain(domain)
            self.domain = domain
            if self.env.need_reset(domain) or evaluation != EvaluationType.NO or epoch == 1:
                obs = self.env.reset(domain=domain, evaluation=evaluation)
                self.update_base_info(obs, domain)

            mb_obs, mb_vae_emb, mb_pre_actions, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], [], [], []
            mb_pi_states = self.pi_states[domain]
            mb_v_states = self.v_states[domain]
            ep_infos = []
            mb_hidden_state = []
            if evaluation == EvaluationType.NO:
                domain_model = None
            else:
                domain_model = self.env.domain

            for _ in range(self.n_steps):
                if self.env.need_reset(domain):
                    self.log_epi_info(evaluation=evaluation)
                    obs = self.env.reset(domain=domain, evaluation=evaluation)
                    self.update_base_info(obs, domain)
                if self.model.vae_handler is not None:
                    code = self.model.vae_handler.embedding(self.obs[domain][:, :self.model.vae_handler.x_shape[0]], deter=evaluation != EvaluationType.NO)
                    self.vae_emb[domain] = np.repeat(code, self.obs[domain].shape[0], axis=0)

                actions, values, self.pi_states[domain], self.v_states[domain], neglogpacs, policy_mean, policy_std = self.model.step(self.obs[domain], self.vae_emb[domain], self.pre_acs[domain],
                                                                           self.pi_states[domain], self.v_states[domain], self.dones[domain], deterministic=deterministic, domain=domain_model)

                assert np.where(np.isnan(actions))[0].shape[0] == 0
                mb_obs.append(self.obs[domain].copy())
                mb_vae_emb.append(self.vae_emb[domain].copy())
                mb_pre_actions.append(self.pre_acs[domain].copy())
                mb_actions.append(actions)
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones[domain].copy())
                clipped_actions = actions
                self.pre_acs[domain] = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
                self.obs[domain][:], rewards, self.dones[domain], infos = self.env.step(clipped_actions)
                self.append_log_info(mb_obs=self.obs[domain], mb_rewards=rewards, info=infos, acs=clipped_actions)
                maybe_ep_info = infos.get('episode')
                mb_hidden_state.append(infos['hidden_state'])
                # if maybe_ep_info is not None:
                #     self.log_epi_info(maybe_ep_info, evaluation)
                mb_rewards.append(rewards)
                self.multi_ts[self.domain] += 1
            if self.env.need_reset(domain):
                self.log_epi_info(evaluation=evaluation)
                obs = self.env.reset(domain=domain, evaluation=evaluation)
                self.update_base_info(obs, domain)
            # assert self.env.need_reset, "{} -> {} {}".format(self.env.selected_env.timestep, self.env.selected_env.T, self.n_steps)
            # compute gamma ret:
            if self.rew_normalize:
                mb_gamma_ret = np.zeros(mb_rewards[0].shape)
                for step in reversed(range(self.n_steps)):
                    mb_gamma_ret = mb_gamma_ret * self.gamma + mb_rewards[step]
                # mb_gamma_ret = swap_and_flatten(mb_gamma_ret)
                self.model.rew_rms.update(mb_gamma_ret.flatten())
            # batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=self.obs[domain].dtype)
            mb_vae_emb = np.asarray(mb_vae_emb)
            mb_pre_actions = np.asarray(mb_pre_actions)
            mb_hidden_state = np.asarray(mb_hidden_state)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            last_values = self.model.value(self.obs[domain], self.vae_emb[domain], self.pre_acs[domain], self.pi_states[domain], self.v_states[domain], self.dones[domain], domain=domain_model)
            true_reward = np.copy(mb_rewards)
            mb_returns = self._compute_gae_ret(mb_rewards, domain, mb_dones, last_values, mb_values)
            mb_obs, mb_vae_emb, mb_pre_actions, mb_hidden_state, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
                map(swap_and_flatten, (mb_obs, mb_vae_emb, mb_pre_actions, mb_hidden_state, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))
            hash = self.env.clustering_hash(mb_obs, mb_hidden_state)
            multi_mb_obs.append(mb_obs)
            multi_mb_vae_emb.append(mb_vae_emb)
            multi_mb_pre_actions.append(mb_pre_actions)
            multi_mb_returns.append(mb_returns)
            multi_mb_dones.append(mb_dones)
            multi_mb_actions.append(mb_actions)
            multi_mb_values.append(mb_values)
            multi_mb_pi_states.append(mb_pi_states)
            multi_mb_v_states.append(mb_v_states)
            multi_ep_infos.append(ep_infos)
            multi_mb_neglogpacs.append(mb_neglogpacs)
            multi_true_reward.append(true_reward)
            multi_cluster_hash.append(hash)
        multi_mb_obs, multi_mb_vae_emb, multi_mb_pre_actions, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
        multi_mb_pi_states,  multi_mb_v_states, multi_ep_infos, multi_mb_neglogpacs, multi_true_reward, \
        = map(self.multi_concat, (multi_mb_obs, multi_mb_vae_emb, multi_mb_pre_actions, multi_mb_returns, multi_mb_dones, multi_mb_actions,
                             multi_mb_values, multi_mb_pi_states, multi_mb_v_states, multi_ep_infos, multi_mb_neglogpacs,  multi_true_reward))
        merge_hash = {}

        id_base = 0
        for hash in multi_cluster_hash:
            for k, v in hash.items():
                if k not in merge_hash:
                    merge_hash[k] = np.array(v) + id_base
                else:
                    merge_hash[k] = np.concatenate([merge_hash[k], np.array(v) + id_base], axis=0)
            id_base += mb_obs.shape[0]
        multi_mb_driver_param_clusters = self.env.hash_to_array(id_base, merge_hash)
        return multi_mb_obs, multi_mb_vae_emb, multi_mb_pre_actions, multi_mb_driver_param_clusters, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
        multi_mb_neglogpacs, multi_mb_pi_states, multi_mb_v_states, multi_ep_infos, multi_true_reward

class EnvAwareRunner(Runner):
    def __init__(self, *args, **kwargs):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super(EnvAwareRunner, self).__init__(*args, **kwargs)
        self.states, self.old_states,self.policy_states, self.pre_obs, self.pre_acs = {}, {}, {}, {}, {}

    def update_base_info(self, obs, domain):
        super(EnvAwareRunner, self).update_base_info(obs, domain)

        self.pre_obs[domain] = self.obs[domain] * 0
        if self.model.initial_state is None:
            self.states[domain] = None
            self.old_states[domain] = None
        else:
            self.states[domain] = np.zeros([self.env.current_num_envs, self.model.initial_state.shape[-1]])  # self.model.initial_state
            self.old_states[domain] = np.zeros([self.env.current_num_envs, self.model.initial_state.shape[-1]])  # self.model.initial_state
        if self.model.policy_initial_state is None:
            self.policy_states[domain] = None
        else:
            self.policy_states[domain] = np.zeros([self.env.current_num_envs, self.model.policy_initial_state.shape[-1]])  # self.model.initial_state
        self.pre_acs[domain] = np.zeros((self.env.current_num_envs,) + self.env.action_space.shape)

    def run(self, epoch, domain=None, evaluation=EvaluationType.NO,
            dynamic_type=DynamicType.POLICY, deterministic=False, one_step=False, keep_did=False, ):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        multi_mb_obs, multi_mb_rewards, multi_mb_actions, multi_mb_values, multi_mb_dones, multi_mb_neglogpacs, \
        multi_mb_pre_obs, multi_mb_pre_actions,  multi_mb_driver_param_clusters, multi_mb_returns, \
        multi_true_reward, multi_mb_states, multi_mb_old_states, multi_mb_policy_states, multi_ep_infos, multi_mb_driver_param_clusters, multi_cluster_hash = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        multi_mb_z_t = []
        self.epoch = epoch
        if evaluation == EvaluationType.NO:
            assert domain is None
            domain_list = self.env.domain_list
            self.n_steps = self.train_steps

        else:
            domain_list = [domain]
            self.n_steps = self.horizon
        self.deterministic = deterministic
        for domain in domain_list:
            self.env.change_domain(domain)
            self.domain = domain
            if self.env.need_reset(domain) or evaluation != EvaluationType.NO or epoch == 1:
                obs = self.env.reset(domain=domain, evaluation=evaluation)
                self.update_base_info(obs, domain)
            # mb stands for minibatch
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_pre_obs, mb_pre_actions, mb_driver_param_clusters =  [], [], [], [], [], [], [], [], []
            mb_z_t = []
            mb_states = self.states[domain]
            mb_old_states = self.old_states[domain]
            mb_policy_states = self.policy_states[domain]
            ep_infos = []
            mb_hidden_state = []
            if evaluation == EvaluationType.NO:
                domain_model = None
            else:
                domain_model = self.env.domain

            for _ in range(self.n_steps):
                if self.env.need_reset(domain):
                    self.log_epi_info(evaluation=evaluation)
                    obs = self.env.reset(domain=domain, evaluation=evaluation)
                    self.update_base_info(obs, domain)
                actions, values, self.states[domain], self.old_states[domain], self.policy_states[domain], neglogpacs, z_t, policy_mean, policy_std = self.model.step(self.pre_obs[domain], self.pre_acs[domain],
                                                                                            self.obs[domain], self.states[domain], self.old_states[domain], self.policy_states[domain],
                                                                                               self.dones[domain], domain=domain_model, deterministic=deterministic,)
                mb_obs.append(self.obs[domain].copy())
                mb_actions.append(actions)
                mb_z_t.append(z_t)
                mb_pre_obs.append(self.pre_obs[domain])
                mb_pre_actions.append(self.pre_acs[domain])
                mb_values.append(values)
                mb_neglogpacs.append(neglogpacs)
                mb_dones.append(self.dones[domain])
                clipped_actions = actions
                self.pre_obs[domain] = self.obs[domain][:]
                self.pre_acs[domain] = actions # self.env.restore_action(actions)
                # Clip the actions to avoid out of bound error
                if isinstance(self.env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
                self.obs[domain][:], rewards, self.dones[domain], infos = self.env.step(clipped_actions)
                mb_hidden_state.append(infos['hidden_state'])
                self.append_log_info(mb_obs=self.obs[domain], mb_rewards=rewards, info=infos, acs=clipped_actions)
                # maybe_ep_info = infos.get('episode')
                # if maybe_ep_info is not None:
                #     self.log_epi_info(maybe_ep_info, evaluation)
                mb_rewards.append(rewards)
                self.multi_ts[self.domain] += 1
            if self.env.need_reset(domain):
                self.log_epi_info(evaluation=evaluation)
                obs = self.env.reset(domain=domain, evaluation=evaluation)
                self.update_base_info(obs, domain)
            # assert self.env.need_reset, "{} -> {} {}".format(self.env.selected_env.timestep, self.env.selected_env.T, self.n_steps)
            # compute gamma ret:
            if self.rew_normalize:
                mb_gamma_ret = np.zeros(mb_rewards[0].shape)
                for step in reversed(range(self.n_steps)):
                    mb_gamma_ret = mb_gamma_ret * self.gamma + mb_rewards[step]
                # mb_gamma_ret = swap_and_flatten(mb_gamma_ret)
                self.model.rew_rms.update(mb_gamma_ret.flatten())

            # batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=self.obs[domain].dtype)
            mb_pre_obs = np.asarray(mb_pre_obs, dtype=self.obs[domain].dtype)
            mb_pre_actions = np.asarray(mb_pre_actions)
            mb_hidden_state = np.asarray(mb_hidden_state)
            mb_z_t = np.asarray(mb_z_t)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)

            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            # update dones
            if self.model.remove_done:
                mb_dones[:] = False
                self.dones[domain][:] = False

            last_values = self.model.value(self.pre_obs[domain], self.pre_acs[domain], self.obs[domain],
                                           self.states[domain], self.old_states[domain], self.policy_states[domain],
                                           self.dones[domain], domain=domain_model) # obs 是最终还未运行的下一个状态。O_{T+1}
            true_reward = np.copy(mb_rewards)
            # discount/bootstrap off value fn
            mb_returns = mb_returns = self._compute_gae_ret(mb_rewards, domain, mb_dones, last_values, mb_values)

            mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_pre_obs, mb_pre_actions, \
            mb_z_t, mb_hidden_state = \
                map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward,
                                       mb_pre_obs, mb_pre_actions, mb_z_t, mb_hidden_state))
            hash = self.env.clustering_hash(mb_obs, mb_hidden_state)
            multi_mb_obs.append(mb_obs)
            multi_mb_z_t.append(mb_z_t)
            multi_mb_returns.append(mb_returns)
            multi_mb_dones.append(mb_dones)
            multi_mb_actions.append(mb_actions)
            multi_mb_values.append(mb_values)
            multi_mb_states.append(mb_states)
            multi_mb_old_states.append(mb_old_states)
            multi_mb_policy_states.append(mb_policy_states)
            multi_ep_infos.append(ep_infos)
            multi_mb_neglogpacs.append(mb_neglogpacs)
            multi_true_reward.append(true_reward)
            multi_mb_pre_obs.append(mb_pre_obs)
            multi_mb_pre_actions.append(mb_pre_actions)
            multi_cluster_hash.append(hash)

        multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
        multi_mb_neglogpacs, multi_mb_states, multi_mb_old_states, multi_mb_policy_states, multi_ep_infos, multi_true_reward, \
        multi_mb_pre_obs, multi_mb_pre_actions, multi_mb_z_t = map(self.multi_concat, (multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
               multi_mb_neglogpacs, multi_mb_states, multi_mb_old_states, multi_mb_policy_states, multi_ep_infos, multi_true_reward,
               multi_mb_pre_obs, multi_mb_pre_actions, multi_mb_z_t))
        merge_hash = {}

        id_base = 0
        for hash in multi_cluster_hash:
            for k, v in hash.items():
                if k not in merge_hash:
                    merge_hash[k] = np.array(v) + id_base
                else:
                    merge_hash[k] = np.concatenate([merge_hash[k], np.array(v) + id_base], axis=0)
            id_base += mb_obs.shape[0]
        multi_mb_driver_param_clusters = self.env.hash_to_array(id_base, merge_hash)

        # # embedding test
        # def map_test():
        #     if self.env.representative_domain.given_scale:
        #         for k, v in merge_hash.items():
        #             dids = v
        #             map_scale = multi_mb_obs[dids][:, np.where(self.env.feature_name == 'std_acs-scale')[0]].astype(
        #                 np.int32)
        #             assert map_scale.max() - map_scale.min() <= 1
        # emb_unit_test_cont.add_test('test_map-{}'.format(evaluation),
        #                             map_test, emb_unit_test_cont.test_type.SINGLE_TIME)
        # emb_unit_test_cont.do_test('test_map-{}'.format(evaluation))
        return multi_mb_obs, multi_mb_returns, multi_mb_dones, multi_mb_actions, multi_mb_values, \
               multi_mb_neglogpacs, multi_mb_states, multi_mb_old_states, multi_mb_policy_states, multi_ep_infos, multi_true_reward, \
               multi_mb_pre_obs, multi_mb_pre_actions,  multi_mb_driver_param_clusters, multi_mb_z_t


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])



def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
