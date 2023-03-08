import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete


from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

import tensorflow_probability as tfp
# from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, ortho_init, _ln
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input

LOG_STD_MAX = 2
LOG_STD_MIN = -20



def lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False, dropout=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param layer_norm: (bool) Whether to apply Layer Normalization or not
    :return: (TensorFlow Tensor) LSTM cell
    """
    _, n_input = [v.value for v in input_tensor[0].get_shape()]
    with tf.variable_scope(scope):
        weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
        weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

        if layer_norm:
            # Gain and bias of layer norm
            gain_x = tf.get_variable("gx", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable("bx", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_h = tf.get_variable("gh", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_h = tf.get_variable("bh", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            gain_c = tf.get_variable("gc", [n_hidden], initializer=tf.constant_initializer(1.0))
            bias_c = tf.get_variable("bc", [n_hidden], initializer=tf.constant_initializer(0.0))

    cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)

    for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
        if dropout:
            _input = tf.layers.dropout(_input, rate=0.5)

        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
                    + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
        in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        if dropout:
            hidden = tf.layers.dropout(hidden, rate=0.5)
        input_tensor[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden]) # last state?
    return input_tensor, cell_state_hidden


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))




def mlp_extractor(flat_observations, net_arch, act_fun, ln=False, res_net=False,
                  latent_value=None):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    if latent_value is None:
        latent_value = latent

    res_net_size = 256
    xp, xv = None, None
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2))
            if ln:
                with tf.variable_scope("pi_{}".format(idx)):
                    latent_policy = tf.contrib.layers.layer_norm(latent_policy, center=True, scale=True)
            latent_policy = act_fun(latent_policy)
            if idx % 2 == 0 and res_net:
                latent_policy = linear(latent_policy, "pi_fc_res_{}".format(idx), res_net_size, init_scale=np.sqrt(2))
                if xp is not None:
                    latent_policy = latent_policy + xp
                xp = latent_policy

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2))
            if ln:
                with tf.variable_scope("vf_{}".format(idx)):
                    latent_value = tf.contrib.layers.layer_norm(latent_value, center=True, scale=True)
            latent_value = act_fun(latent_value)
            if idx % 2 == 0 and res_net:
                latent_value = linear(latent_value, "vf_fc_res_{}".format(idx), res_net_size, init_scale=np.sqrt(2))
                if xv is not None:
                    latent_value = latent_value + xv
                xv = latent_value

    return latent_policy, latent_value



class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="action_ph")
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitely (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitely)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 *args, **kwargs):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale, *args, **kwargs)
        #self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

    #@property
    #def pdtype(self):
    #    """ProbabilityDistributionType: type of the distribution for stochastic actions."""
    #    return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 state_shape, reuse=False, scale=False, *args, **kwargs):
        super(RecurrentActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                   n_batch, reuse=reuse, scale=scale, *args, **kwargs)

        with tf.variable_scope("input", reuse=False):
            self._dones_ph = tf.placeholder(tf.float32, (n_batch, ), name="dones_ph")  # (done t-1)
            state_ph_shape = (self.n_env, ) + tuple(state_shape)
            self._states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

        initial_state_shape = (self.n_env, ) + tuple(state_shape)
        self._initial_state = np.zeros(initial_state_shape) # np.zeros(initial_state_shape, dtype=np.float32)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class EnvAwarePolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", name='env_aware_pi', learn_var=False,
                 stop_critic_gradient=False, layer_norm=False, res_net=False, lstm_policy=False,
                 lstm_critic=False, n_lstm=None, redun_distri_info=False, dropout=False,
                 processed_obs=None, env_parameter_input=None, old_env_parameter_input=None, state_distribution_emb=None, **kwargs):

        self.layers = layers
        self.name = name
        self.lstm_critic = lstm_critic
        self.net_arch = net_arch
        self.act_fun = act_fun
        self.dropout = dropout
        self.lstm_policy = lstm_policy
        self.feature_extraction = feature_extraction
        self.redun_distri_info = redun_distri_info

        with tf.variable_scope(self.name, reuse=reuse):
            super(EnvAwarePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                    scale=(feature_extraction == "cnn"))
            with tf.variable_scope("input", reuse=True):
                self.dones_ph = tf.placeholder(tf.float32, (n_batch,), name="dones_ph")  # (done t-1)
                state_ph_shape = (self.n_env,) + tuple((2 * n_lstm,))
                self.states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

            if processed_obs is not None:
                self._processed_obs = processed_obs
            self.env_parameter_input = env_parameter_input
            self.old_env_parameter_input = old_env_parameter_input
            self.state_distribution_emb = state_distribution_emb
            if state_distribution_emb is not None and redun_distri_info:
                state_distribution_emb_repeat = state_distribution_emb # tf.repeat(tf.expand_dims(state_distribution_emb, 0), tf.shape(self._processed_obs)[0], axis=0)
                obs_with_env_param = tf.concat([self.processed_obs, env_parameter_input, state_distribution_emb_repeat], axis=-1)
            elif redun_distri_info:
                obs_with_env_param = tf.concat([self.processed_obs, env_parameter_input], axis=-1)
            else:
                print('env_parameter_input', env_parameter_input)
                obs_with_env_param = tf.concat([env_parameter_input], axis=-1)
            if self.layers is not None:
                warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                              "(it has a different semantics though).", DeprecationWarning)
                if self.net_arch is not None:
                    warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                                  DeprecationWarning)

            if self.net_arch is None:
                if self.layers is None:
                    self.layers = [64, 64]
                self.net_arch = [dict(vf=self.layers, pi=self.layers)]

                if self.feature_extraction == "cnn":
                    raise NotImplementedError()
                flatten_input = tf.layers.flatten(obs_with_env_param)
                flatten_input_critic = old_env_parameter_input
                if stop_critic_gradient:
                    latent_critic = tf.stop_gradient(flatten_input_critic)
                else:
                    latent_critic = flatten_input_critic
                if state_distribution_emb is not None:
                    latent_critic = tf.concat([latent_critic, self.processed_obs, state_distribution_emb], axis=-1)
                else:
                    latent_critic = tf.concat([latent_critic, self.processed_obs], axis=-1)

                if lstm_critic:
                    input_sequence = batch_to_seq(latent_critic, self.n_env, n_steps)
                    masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                    rnn_output, self.snew_c = lstm(input_sequence, masks, self.states_ph, 'lstm_c', n_hidden=n_lstm, dropout=self.dropout)
                    rnn_output = seq_to_batch(rnn_output)
                    if redun_distri_info:
                        latent_critic = tf.concat([latent_critic, rnn_output], axis=-1)
                    else:
                        latent_critic = rnn_output
                if lstm_policy:
                    input_sequence = batch_to_seq(flatten_input, self.n_env, n_steps)
                    masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                    rnn_output, self.snew_p = lstm(input_sequence, masks, self.states_ph, 'lstm_p', n_hidden=n_lstm, dropout=self.dropout)
                    rnn_output = seq_to_batch(rnn_output)
                    if redun_distri_info:
                        flatten_input = tf.concat([flatten_input, rnn_output], axis=-1)
                    else:
                        flatten_input = rnn_output

                pi_latent, vf_latent = mlp_extractor(flatten_input,
                                                     self.net_arch, self.act_fun,
                                                     ln=layer_norm, res_net=res_net, latent_value=latent_critic)
                self._value_fn = linear(vf_latent, 'vf', 1)
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01, act_fn=tf.nn.tanh,
                                                               learn_var=learn_var)
        initial_state_shape = (self.n_env, ) + tuple((2 * n_lstm,))
        self._initial_state = np.zeros(initial_state_shape) # np.zeros(initial_state_shape, dtype=np.float32)
        self._setup_init()
        self._obs_ph = self._processed_obs


    @property
    def initial_state(self):
        return self._initial_state

    def step(self, obs, env_param=None, old_env_param=None, state_distribution_emb=None, policy_states=None, mask=None, deterministic=False):
        feed_dict = {self._processed_obs: obs, self.env_parameter_input: env_param, self.old_env_parameter_input: old_env_param}
        if self.state_distribution_emb is not None:
            feed_dict[self.state_distribution_emb] = state_distribution_emb
        feed_dict[self.dones_ph] = mask
        if deterministic:
            ops = [self.deterministic_action, self.value_flat, self.neglogp]
        else:
            ops =[self.action, self.value_flat, self.neglogp]
        if self.lstm_critic:
            feed_dict[self.states_ph] = policy_states
            ops += [self.snew]
            action, value, neglogp, state = self.sess.run(ops, feed_dict)
            return action, value, state, neglogp
        else:
            action, value, neglogp = self.sess.run(ops, feed_dict)
            return action, value, self.initial_state, neglogp

    def proba_step(self, obs, env_param=None, old_env_param=None, policy_states=None, state_distribution_emb=None, mask=None):
        feed_dict = {self.obs_ph: obs, self.dones_ph: mask, self.env_parameter_input: env_param, self.old_env_parameter_input: old_env_param}
        if self.lstm_critic:
            feed_dict[self.states_ph] = policy_states
        if self.state_distribution_emb is not None and self.redun_distri_info:
            feed_dict[self.state_distribution_emb] = state_distribution_emb
        return self.sess.run(self.policy_proba, feed_dict)

    def value(self, obs, env_param=None,  old_env_param=None, policy_states=None, state_distribution_emb=None, mask=None):
        feed_dict = {self.obs_ph: obs, self.env_parameter_input: env_param, self.dones_ph: mask, self.old_env_parameter_input: old_env_param}
        if self.lstm_critic:
            feed_dict[self.states_ph] = policy_states
        if self.state_distribution_emb is not None:
            feed_dict[self.state_distribution_emb] = state_distribution_emb
        return self.sess.run(self.value_flat, feed_dict)


class EnvExtractorPolicy(RecurrentActorCriticPolicy):
    recurrent = True
    def __init__(self, sess, ob_space, ac_space, total_env,  n_steps, n_batch, n_lstm=256, reuse=False,
                 state_distribution_emb=None, use_resnet=True, consistent_ecoff=0, layers=None, learn_var=False, name='env_extractor',
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, init_scale_output=1.0,
                 ent_out_act=tf.nn.tanh, dropout=False,
                 layer_norm=False, lstm_layer_norm=False,  feature_extraction="mlp",
                 **kwargs):
        self.total_env = total_env
        self.feature_extraction = feature_extraction
        self.net_arch = net_arch
        self.layers = layers
        self.n_lstm = n_lstm
        self.layer_norm = layer_norm
        self.lstm_layer_norm = lstm_layer_norm
        self.learn_var = learn_var
        self.name = name
        self.act_fun = act_fun
        self.ent_out_act = ent_out_act
        self._kwargs_check(feature_extraction, kwargs)
        # from stable_baselines.a2c.utils import linear
        if self.feature_extraction == "cnn":
            raise NotImplementedError()
        with tf.variable_scope(self.name, reuse=reuse):
            # state_shape = [n_lstm * 2] dim because of the cell and hidden state of the LSTM
            super(EnvExtractorPolicy, self).__init__(sess, ob_space, ac_space, total_env, n_steps, n_batch,
                                                     state_shape=(2 * n_lstm,), reuse=False,
                                                     scale=(feature_extraction == "mlp"), add_action_ph=True)

            latent = tf.layers.flatten(self.processed_obs)
            # latent_acs = tf.layers.flatten(self.action_ph)
            self.state_distribution_emb = state_distribution_emb
            state_distribution_emb_repeat = state_distribution_emb # tf.repeat(tf.expand_dims(state_distribution_emb, 0), tf.shape(latent)[0], axis=0)
            latent = tf.concat([latent, state_distribution_emb_repeat], axis=-1)
            policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
            value_only_layers = []  # Layer sizes of the network that only belongs to the value network
            if self.layers is None:
                layers = [128, 128]
            else:
                layers = self.layers
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(latent, **kwargs)
            else:
                extracted_features = tf.layers.flatten(latent)
                for i, layer_size in enumerate(layers):
                    extracted_features = linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                        init_scale=np.sqrt(2))
                    if self.layer_norm:
                        extracted_features = tf.contrib.layers.layer_norm(extracted_features, center=True, scale=True)
                    extracted_features = act_fun(extracted_features)
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=self.n_lstm,
                                             layer_norm=self.lstm_layer_norm, dropout=dropout)
            rnn_output = seq_to_batch(rnn_output)
            pi_latent_vector = rnn_output
            # Build the non-shared part of the policy-network
            # latent_policy = latent
            # TODO: why not init_scale = 0.001 here like in the feedforward
            self.acs_size = self.pdtype.size
            self.mean = self.ent_out_act(linear(pi_latent_vector, 'pi', self.acs_size, init_scale=init_scale_output, init_bias=0.))
            # TODO learable var
            # logstd = tf.get_variable(name='pi/logstd', shape=[1, self.acs_size], initializer=tf.zeros_initializer())
            # logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
            # std = tf.exp(logstd) + self.mean * 0.0
            # self._proba_distribution = tfp.distributions.MultivariateNormalDiag(self.mean, std)
            self._policy = self.mean
            self._action = self.mean  # self.proba_distribution.sample()
            self._deterministic_action = self.mean  # self.proba_distribution.mode()
            # self._neglogp = self.proba_distribution.log_prob(self.action) * -1
            # self._policy_proba = [self.mean, std]
            self._obs_ph = self._processed_obs
            # self._proba_distribution, self._policy, self.q_value = \
            #     self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        # self._setup_init()


    def step(self, pre_obs_acs, state=None, dis_emb=None, mask=None, deterministic=True):
        assert deterministic
        if deterministic:
            return self.sess.run([self.deterministic_action, self.snew],
                                 {self.obs_ph: pre_obs_acs,  self.states_ph: state,
                                  self.state_distribution_emb: dis_emb,
                                  self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.snew],
                                 {self.obs_ph: pre_obs_acs, self.states_ph: state,
                                  self.state_distribution_emb: dis_emb,
                                  self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError
        # return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        raise NotImplementedError
        # return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 dropout=False,
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, state_shape=(2 * n_lstm, ),
                                         reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm, dropout=dropout)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)
            # , act_fn=tf.nn.tanh
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="mlp", **_kwargs)


_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
