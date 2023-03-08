import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
import tensorflow_probability as tfp

from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

from lts.src.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
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
        weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=tf.orthogonal_initializer(init_scale))
        weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=tf.orthogonal_initializer(init_scale))
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


def res_lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False, use_resnet=False,
             consistent_ecoff=0):
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
        last_hidden = hidden
        for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
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
                # cell_state = tf.contrib.layers.layer_norm(cell_state, center=True, scale=True)
                hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
            else:
                hidden = out_gate * tf.tanh(cell_state)
            if use_resnet:
                hidden = hidden + consistent_ecoff * last_hidden
                last_hidden = hidden
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
                print('self._obs_ph', self._obs_ph)
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
        self._pdtype = make_proba_dist_type(ac_space)
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

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

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
            # state_ph_shape = (self.n_env, ) + tuple(state_shape)
            # self._states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

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

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


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

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None, name='lstm_policy',
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, lstm_layer_norm=False, post_policy_layers=None,
                 dropout=False, redun_info=False, stop_critic_gradient=False, no_share_layer=False,
                 feature_extraction="cnn", oc=False, mask_len=0,
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, state_shape=(2 * n_lstm, ),
                                         reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.name = name
        self.snew = None
        self.stop_critic_gradient = stop_critic_gradient
        self.no_share_layer = no_share_layer
        state_ph_shape = (self.n_env,) + tuple((2 * n_lstm, ))
        with tf.variable_scope("input", reuse=False):
            if self.no_share_layer:
                self.pi_states_ph = tf.placeholder(tf.float32, state_ph_shape, name="pi_states_ph")
                self.v_states_ph = tf.placeholder(tf.float32, state_ph_shape, name="v_states_ph")
            else:
                self.pi_states_ph = self.v_states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")
        with tf.variable_scope(self.name, reuse=reuse):
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
                        net_arch = [dict(vf=layers, pi=layers)]
                        if self.no_share_layer:
                            if oc:
                                range = tf.range(0, extracted_features.shape[1], 1)
                                range_row = tf.expand_dims(range, 0)
                                mask = tf.greater(range_row, mask_len - 1)
                                mask = tf.cast(mask, tf.float32)
                                p_input = extracted_features * mask
                            else:
                                p_input = extracted_features
                            pi_latent, vf_latent = mlp_extractor(p_input, net_arch, act_fun,
                                                                 ln=layer_norm, res_net=False,
                                                                 latent_value=extracted_features)
                            input_sequence = batch_to_seq(pi_latent, self.n_env, n_steps)
                            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                            pi_rnn_output, self.pi_snew = lstm(input_sequence, masks, self.pi_states_ph, 'lstm_pi',
                                                         n_hidden=n_lstm,
                                                         layer_norm=lstm_layer_norm, dropout=dropout)
                            pi_rnn_output = seq_to_batch(pi_rnn_output)
                            self.pi_rnn_output = pi_rnn_output

                            input_sequence = batch_to_seq(vf_latent, self.n_env, n_steps)
                            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                            v_rnn_output, self.v_snew = lstm(input_sequence, masks, self.v_states_ph, 'lstm_v',
                                                               n_hidden=n_lstm,
                                                               layer_norm=lstm_layer_norm, dropout=dropout)
                            v_rnn_output = seq_to_batch(v_rnn_output)
                            self.v_rnn_output = v_rnn_output
                        else:
                            for i, layer_size in enumerate(layers):
                                extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                                    init_scale=np.sqrt(2)))

                            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                            rnn_output, snew = lstm(input_sequence, masks, self.pi_states_ph, 'lstm', n_hidden=n_lstm,
                                                         layer_norm=lstm_layer_norm, dropout=dropout)
                            rnn_output = seq_to_batch(rnn_output)
                            self.pi_snew = self.v_snew = snew
                            self.pi_rnn_output = self.v_rnn_output = pi_rnn_output = v_rnn_output = rnn_output
                    if post_policy_layers is []:
                        pi_latent = pi_rnn_output
                        vf_latent = v_rnn_output
                    else:
                        post_net_arch = [dict(vf=post_policy_layers, pi=post_policy_layers)]
                        if redun_info:
                            pi_latent = tf.concat([self.processed_obs, pi_rnn_output], axis=-1)
                            vf_latent = tf.concat([self.processed_obs, v_rnn_output], axis=-1)
                        with tf.variable_scope("post", reuse=reuse):
                            pi_latent, vf_latent = mlp_extractor(pi_latent, post_net_arch, act_fun,
                                                                 ln=layer_norm, res_net=False, latent_value=vf_latent)

                    value_fn = linear(vf_latent, 'vf', 1)

                    self._proba_distribution, self._policy, self.q_value = \
                        self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, act_fn=tf.nn.tanh)

                self._value_fn = value_fn
            self._setup_init()

    def step(self, obs, pi_state=None, v_state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.pi_snew, self.v_snew, self.neglogp],
                                 {self.obs_ph: obs, self.pi_states_ph: pi_state,
                                  self.v_states_ph: v_state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.pi_snew, self.v_snew, self.neglogp],
                                 {self.obs_ph: obs, self.pi_states_ph: pi_state,
                                  self.v_states_ph: v_state, self.dones_ph: mask})

    def proba_step(self, obs, pi_state=None, v_state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.pi_states_ph: pi_state,
                                                 self.v_states_ph: v_state, self.dones_ph: mask})

    def value(self, obs, pi_state=None, v_state=None,  mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.pi_states_ph: pi_state,
                                               self.v_states_ph: v_state, self.dones_ph: mask})


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
                 learn_var=False, name='ffp', layer_norm=False, res_net=False, init_scale=0.01,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.name = name
        self.res_net = res_net
        self.layer_norm = layer_norm
        self.initial_state = np.array([None])
        with tf.variable_scope(self.name, reuse=reuse):
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
                    pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun,
                                                         ln=self.layer_norm, res_net=self.res_net)

                self._value_fn = linear(vf_latent, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, act_fn=tf.nn.tanh,
                                                               learn_var=learn_var)

            self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        obs = np.expand_dims(obs, axis=0)
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        obs = np.expand_dims(obs, axis=0)
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
