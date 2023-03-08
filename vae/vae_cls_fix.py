import tensorflow as tf
from common.dataset import Dset
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# tfd = tf.contrib.distributions
from tensorflow_probability import distributions as tfd
from sklearn.neighbors import KernelDensity
# from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from common.tester import tester
from common import logger
from common.optimizers import RectifiedAdam# , RAdamOptimizer
# from keras_radam.training import RAdamOptimizer
from common.config import *
from common.utils import *
import pickle

def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
        axis='both', which='both', left='off', bottom='off',
        labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
    for index, sample in enumerate(samples):
        ax[index].imshow(sample, cmap='gray')
        ax[index].axis('off')


class VAE(object):
    GAUSSIAN = 'gau'
    INDEPENDENT_NORM = 'independent_norm'
    GAUSSIAN_FULLRANK= 'gau_full'
    BERNOULLI = 'ber'
    LOG_GAUSSIAN = 'log-gau'
    TRUN_NORMAL = 'tru-norm'

    def __init__(self, hid_size, hid_layers, x_shape, lr, z_size, decode_type, low_bound=None, upper_bound=None,
                 log_norm_index=None, discrete_index=None, max_category=None, normal_distribution_index=None,
                 condition_distribution_map_index=None, remove_min_index=None, condition_index_bound_val=None,
                 layer_filter_index=None, layer_norm=False, share_codes=None, merge_dim_likelihood=False,
                 prior_data_index=[], optimizer=None, split_condition_predict=None, split_z=None,
                 weight_loss=False, zero_init=False, scale_hid=1, split_z_by_layer=False, constant_z_scale=False,
                 vae_layer_index=-1, vae_layer_number=1, cond_embedding_layer=0, cond_with_gau=False,
                 full_rank_gau=False, name=''):
        self.hid_size = hid_size
        self.weight_loss = weight_loss
        self.cond_with_gau = cond_with_gau
        self.constant_z_scale = constant_z_scale
        self.cond_embedding_layer = cond_embedding_layer
        self.zero_init = zero_init
        self.split_condition_predict = split_condition_predict
        self.vae_layer_index= vae_layer_index
        self.vae_layer_number = vae_layer_number
        self.split_z = split_z
        self.split_z_by_layer = split_z_by_layer
        self.scale_hid = scale_hid
        self.x_shape = x_shape
        self.max_category = int(max_category)
        self.z_size = z_size
        self.hid_layers = hid_layers
        self.lr = lr
        self.merge_dim_likelihood = merge_dim_likelihood
        self.decode_type = decode_type
        self.low_bound = low_bound
        self.upper_bound = upper_bound
        self.layer_norm = layer_norm
        self.optimizer = optimizer
        self.full_rank_gau = full_rank_gau
        self.reuse_encoder = False
        self.share_codes = share_codes
        self.prior_data_index = np.sort(np.array(prior_data_index))
        # feature belong to this layer
        self.layer_filter_index = np.sort(np.array(layer_filter_index))

        def feature_index_filter(index_list):
            if self.layer_filter_index is None:
                return np.sort(np.array(index_list))
            else:
                return np.sort(np.array([x for x in index_list if x in self.layer_filter_index]))

        # 4 type distribution
        self.discrete_index = feature_index_filter(discrete_index)
        self.log_norm_index = feature_index_filter(log_norm_index)
        self.normal_index = feature_index_filter(normal_distribution_index)
        self.condition_distribution_map_index = feature_index_filter(condition_distribution_map_index)
        self.remove_min_index = feature_index_filter(remove_min_index)
        self.condition_index_bound_val = condition_index_bound_val

        def condition_map(index_list, condition_pass):
            if self.condition_distribution_map_index.shape[0] == 0:
                return np.array([]) if condition_pass else index_list
            else:
                if condition_pass:
                    return np.sort(np.array([x for x in index_list if x in self.remove_min_index]))
                else:
                    return np.sort(np.array([x for x in index_list if x not in self.remove_min_index]))

        # condition map:
        assert not self.split_condition_predict
        if self.split_condition_predict:
            self.log_norm_cond_index = condition_map(self.log_norm_index, True)
            if not self.full_rank_gau:
                self.normal_cond_index = condition_map(self.normal_index, True)
            self.discrete_cond_index = condition_map(self.discrete_index, True)

            self.log_norm_index = condition_map(self.log_norm_index, False)
            if not self.full_rank_gau:
                self.normal_index = condition_map(self.normal_index, False)
            else:
                self.normal_cond_index = self.discrete_cond_index = np.array([])
            self.discrete_index = condition_map(self.discrete_index, False)
        else:
            self.log_norm_cond_index = self.normal_cond_index = self.discrete_cond_index = np.array([])
        self.name = name

        assert len(self.x_shape) == 1

        # if log_norm_index is not None:
        #     self.layer_filter_index = np.where(layer_filter_index == 1)[0]
        #     self.log_norm_index = np.where(log_norm_index == 1)[0] & self.layer_filter_index
        #     self.discrete_index = np.where(discrete_index == 1)[0] & self.layer_filter_index
        #     self.normal_distribution_index = np.where(normal_distribution_index == 1)[0] & self.layer_filter_index
        #     if condition_distribution_map_index is not None:
        #         self.condition_distribution_map_index = np.where(condition_distribution_map_index == 1)[0] & self.layer_filter_index
        #     else:
        #         self.condition_distribution_map_index = np.zeros(x_shape)
        #     self.max_category = 10
        # else:
        #     self.log_norm_index = np.ones(x_shape)
        #     self.discrete_index = np.zeros(x_shape)
        #     self.normal_distribution_index = np.zeros(x_shape)
        #     self.normal_distribution_index = np.zeros(x_shape)
        #     # self.bernoulli_distribution_index = np.zeros(x_shape)
        #     self.condition_distribution_map_index = np.zeros(x_shape)
        #     self.max_category = max_category
        # assert len(dataset.shape) == 2
        # self.X_dim = dataset.shape[1:]
        # self.h_dim = h_dim
        # self.z_dim = z_dim
        # self.batch_size = batch_size

    def cast_remove_min_index_to_condition_index(self, remove_min_index):
        return self.condition_distribution_map_index[np.where(np.isin(self.remove_min_index, remove_min_index))]

    @property
    def layer_vae_feature_len(self):
        return self.layer_filter_index.shape[0]

    @property
    def layer_prior_len(self):
        return self.prior_data_index.shape[0]

    def construction(self):
        max_category = self.max_category
        def make_encoder(data, code_size, name='encoder'):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # calculate multi gaussian: http://www.tina-vision.net/docs/memos/2003-003.pdf
                # https://github.com/katerakelly/oyster/blob/cd09c1ae0e69537ca83004ca569574ea80cf3b9c/rlkit/torch/sac/agent.py#L10
                with tf.variable_scope('emb'):
                    x = tf.layers.flatten(data)
                    o_x = x = tf.layers.dense(x, self.hid_size, tf.nn.leaky_relu)
                    for _ in range(self.hid_layers - 1):
                        x = tf.layers.dense(x, self.hid_size)
                        if self.layer_norm:
                            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                        x = tf.nn.leaky_relu(x)
                    x = o_x + x
                locs = tf.layers.dense(x, code_size, name='loc')
                if self.constant_z_scale:
                    scales = locs * 0 + 1
                else:
                    scales = tf.layers.dense(x, code_size, tf.nn.softplus, name='scale')
                # joint prob
                scales_square = tf.square(scales) + 1e-7
                scale_square = 1 / tf.reduce_sum(1 / scales_square, axis=-2)
                loc = scale_square * tf.reduce_sum(locs / scales_square, axis=-2)
                scale = tf.sqrt(scale_square)
                # scale_2 = tf.square(scale)
                # loc = tf.reduce_sum(loc * scale_2, axis=0, keepdims=True) / tf.reduce_sum(scale_2, axis=0, keepdims=True)
                # scale = tf.reduce_sum(1 / scale_2, axis=0, keepdims=True)
                return tfd.Normal(loc, scale)

        def make_prior(code_size):
            loc = tf.zeros(code_size, dtype=tf.float32)
            scale = tf.ones(code_size, dtype=tf.float32)
            return tfd.Normal(loc, scale)

        def make_decoder(code, data_shape, decode_type, code_start_index, name='', ret_param=False):
            with tf.variable_scope(decode_type + '-' + name, reuse=tf.AUTO_REUSE):
                data_size = np.prod(data_shape)
                # x = code
                with tf.variable_scope('emb'):
                    if self.split_z:
                        select_index = list(
                            range(code_start_index * self.code_part_len, (code_start_index + 1) * self.code_part_len))
                        select_index += list(range(self.z_size, int(code.shape[-1])))
                        x = tf.gather(code, indices=select_index, axis=-1)
                    elif self.split_z_by_layer:
                        code_per_layer = int(self.z_size / self.vae_layer_number)
                        select_index = list(range(self.vae_layer_index * code_per_layer,
                                                  min((self.vae_layer_index + 1) * code_per_layer, self.z_size)))
                        select_index += list(range(self.z_size, int(code.shape[-1])))
                        x = tf.gather(code, indices=select_index, axis=-1)
                    else:
                        x = code
                    o_x = x = tf.layers.dense(x, self.hid_size, tf.nn.leaky_relu)
                    for _ in range(self.hid_layers - 1):
                        x = tf.layers.dense(x, self.hid_size)
                        if self.layer_norm:
                            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                        x = tf.nn.leaky_relu(x)
                    x = x + o_x
                if decode_type is self.BERNOULLI:
                    logit = tf.layers.dense(x, np.prod(data_shape))
                    logit = tf.reshape(logit, [-1] + data_shape)
                    # 2: 最后维度被认为是event dims:
                    distribution = tfd.Bernoulli(logit, allow_nan_stats=False)
                    dis_param = (logit)  # tfd.Independent(tfd.Bernoulli(logit), 2)
                elif decode_type is self.GAUSSIAN:
                    loc = tf.layers.dense(x, data_size, name='loc')
                    scale = tf.layers.dense(x, data_size, tf.nn.softplus, name='scale')
                    loc = tf.reshape(loc, [-1] + data_shape)
                    scale = tf.reshape(scale, [-1] + data_shape)
                    distribution = tfd.MultivariateNormalDiag(loc, scale,
                                                      allow_nan_stats=False)
                    dis_param = (loc, scale)  # tfd.MultivariateNormalDiag(loc, scale, allow_nan_stats=False) # tfd.Independent(, len(data_shape)-1)
                elif decode_type is self.INDEPENDENT_NORM:
                    loc = tf.layers.dense(x, data_size, name='loc')
                    scale = tf.layers.dense(x, data_size, tf.nn.softplus, name='scale')
                    loc = tf.reshape(loc, [-1] + data_shape)
                    scale = tf.reshape(scale, [-1] + data_shape)
                    distribution =  tfd.Normal(loc, scale,
                                      allow_nan_stats=False)
                    dis_param = (loc, scale)   # tfd.MultivariateNormalDiag(loc, scale, allow_nan_stats=False) # tfd.Independent(, len(data_shape)-1)
                elif decode_type is self.GAUSSIAN_FULLRANK:
                    loc = tf.layers.dense(x, data_size, name='loc', kernel_initializer=tf.initializers.zeros(),
                                          bias_initializer=tf.initializers.zeros())
                    assert self.zero_init
                    o_x_scale = x_scale = tf.layers.dense(x, self.hid_size * 2, tf.nn.leaky_relu)

                    for hid_i in range(self.scale_hid - 1):
                        x_scale = tf.layers.dense(x_scale, self.hid_size * 2, name='scale-hid-{}'.format(hid_i),
                                                 kernel_initializer=tf.initializers.glorot_normal())
                        if self.layer_norm:
                            x_scale = tf.contrib.layers.layer_norm(x_scale, center=True, scale=True)
                        x_scale = tf.nn.leaky_relu(x_scale)
                    x_scale = x_scale + o_x_scale # TODO if self.scale_hid > 1
                    # TODO: scale [-1, 1], 允许表达 负相关的关系
                    scale = tf.layers.dense(x_scale, int((1 + data_size) * data_size / 2), name='scale', kernel_initializer=tf.initializers.zeros(),
                                            bias_initializer=tf.initializers.zeros(), activation=tf.nn.tanh) + 1
                    scale = tf.log(scale + 1e-5)
                    loc = tf.reshape(loc, [-1] + data_shape)
                    # scale = tf.reshape(scale, [-1] + data_shape + data_shape)
                    scale = tfd.fill_triangular(scale) + tf.expand_dims(tf.diag(tf.exp(tf.ones(data_size)) * 0.5), axis=0)
                    chol = tfd.matrix_diag_transform(scale, transform=tf.nn.softplus)
                    distribution = tfd.MultivariateNormalTriL(loc, chol, validate_args=True,
                                      allow_nan_stats=False)
                    dis_param = (loc, chol)  # tfd.MultivariateNormalDiag(loc, scale, allow_nan_stats=False) # tfd.Independent(, len(data_shape)-1)

                elif decode_type is self.LOG_GAUSSIAN:
                    loc = tf.layers.dense(x, data_size, name='loc')
                    scale = tf.layers.dense(x, data_size, tf.nn.softplus, name='scale')
                    loc = tf.reshape(loc, [-1] + data_shape)
                    scale = tf.reshape(scale, [-1] + data_shape)
                    distribution = tfd.LogNormal(loc, scale, validate_args=True, allow_nan_stats=False)
                    dis_param = (loc, scale)
                elif decode_type is self.TRUN_NORMAL:
                    loc = tf.layers.dense(x, data_size, name='loc')
                    scale = tf.layers.dense(x, data_size, tf.nn.softplus, name='scale')
                    loc = tf.reshape(loc, [-1] + data_shape)
                    scale = tf.reshape(scale, [-1] + data_shape)
                    distribution = tfd.TruncatedNormal(loc=loc, scale=scale, low=self.low_bound, high=self.upper_bound,
                                               validate_args=True, allow_nan_stats=False)
                    dis_param = (loc, scale)

                else:
                    raise NotImplementedError
                if ret_param:
                    return distribution, dis_param
                else:
                    return distribution

        def tensor_indices(data, indices):
            return tf.transpose(
                tf.gather_nd(tf.transpose(data), indices=np.expand_dims(indices, axis=1), batch_dims=0, ))

        def discrete_decoder(code, data_shape, code_start_index, name='', ret_param=False):
            with tf.variable_scope('disc' + name, reuse=tf.AUTO_REUSE):
                data_size = np.prod(data_shape)
                with tf.variable_scope('emb'):
                    if self.split_z:
                        select_index = list(
                            range(code_start_index * self.code_part_len, (code_start_index + 1) * self.code_part_len))
                        select_index += list(range(self.z_size, int(code.shape[-1])))
                        x = tf.gather(code, indices=select_index, axis=-1)
                    elif self.split_z_by_layer:
                        code_per_layer = int(self.z_size / self.vae_layer_number)
                        select_index = list(range(self.vae_layer_index * code_per_layer, min((self.vae_layer_index + 1) * code_per_layer, self.z_size)))
                        select_index += list(range(self.z_size, int(code.shape[-1])))
                        x = tf.gather(code, indices=select_index, axis=-1)
                    else:
                        x = code
                    for _ in range(self.hid_layers):
                        x = tf.layers.dense(x, self.hid_size)
                        if self.layer_norm:
                            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
                        x = tf.nn.leaky_relu(x)
                logits = tf.layers.dense(x, (max_category) * data_size, tf.nn.softplus)
                logits = tf.reshape(logits, [-1] + data_shape + [max_category])
                if ret_param:
                    return tfd.Categorical(logits=logits, validate_args=True, allow_nan_stats=False), (logits)
                else:
                    return tfd.Categorical(logits=logits, validate_args=True, allow_nan_stats=False)

        def cond_embedding(cond_var):
            with tf.variable_scope('cond_emb', reuse=tf.AUTO_REUSE):

                for hid_i in range(self.cond_embedding_layer):
                    cond_var = tf.layers.dense(cond_var, 64, name='cond_embedding-{}'.format(hid_i),
                                              kernel_initializer=tf.initializers.glorot_normal())
                    if self.layer_norm:
                        cond_var = tf.contrib.layers.layer_norm(cond_var, center=True, scale=True)
                    cond_var = tf.nn.tanh(cond_var)
                return cond_var

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            def safe_log(ops):
                return tf.math.log(ops + 1e-6)

            self.to_merge_encode_samples_ops = []
            self.to_merge_encode_samples_likelihood = []
            self.to_merge_encode_samples_name = []
            self.to_merge_encode_samples_idx = []
            self.code_part_len = int(self.z_size / 4)
            sample_number = tf.placeholder(tf.int32, [])
            # make_encoder = tf.make_template('encoder', make_encoder)
            # data shape is useless to be a parameters. make_template only allow ops with the same data shape.
            make_log_norm_decoder = lambda code, data_shape, name='', ret_param=False: make_decoder(code, data_shape, self.LOG_GAUSSIAN,
                                                                                   code_start_index=0, name=name,
                                                                                   ret_param=ret_param)
            if self.full_rank_gau:
                make_normal_decoder = lambda code, data_shape, name='', ret_param=False: make_decoder(code, data_shape,
                                                                                           self.GAUSSIAN_FULLRANK,
                                                                                           code_start_index=1, name=name,
                                                                                   ret_param=ret_param)
            else:
                make_normal_decoder = lambda code, data_shape, name='', ret_param=False: make_decoder(code, data_shape,
                                                                                           self.INDEPENDENT_NORM,
                                                                                           code_start_index=1, name=name,
                                                                                   ret_param=ret_param)
            make_bernoulli_decoder = lambda code, data_shape, name='', ret_param=False: make_decoder(code, data_shape, self.BERNOULLI,
                                                                                    code_start_index=2, name=name,
                                                                                   ret_param=ret_param)
            make_discrete_decoder = lambda code, data_shape, name='', ret_param=False: discrete_decoder(code, data_shape,
                                                                                       code_start_index=3, name=name,
                                                                                   ret_param=ret_param)
            make_cond_embedding = lambda cond_var: cond_embedding(cond_var)
            with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
                # Define the model.
                if self.share_codes is not None:
                    prior = self.share_codes['prior']
                    posterior = self.share_codes['posterior']  # make_encoder(data, code_size=self.z_size)
                    data = self.share_codes['data']
                    code = self.share_codes['code']
                    code_mode = self.share_codes['code_mode']
                    self.reuse_encoder = True
                else:
                    data = tf.placeholder(tf.float32, [None, ] + self.x_shape, name='data_ph')
                    prior = make_prior(code_size=self.z_size)
                    posterior = make_encoder(data, code_size=self.z_size)
                    code = posterior.sample()
                    code_mode = posterior.mode()
                    self.share_codes = {"prior": prior, "posterior": posterior, "data": data,
                                        "code": code, "code_mode": code_mode}
                    self.reuse_encoder = False

                # Define the loss.MultivariateNormalDiag
                code = tf.expand_dims(code, 0)  # TODO: don't use expand_dims
                divergence = tfd.kl_divergence(posterior, prior)

                # add prior
                if self.layer_prior_len > 0:
                    # when training, prior data should has the same order to data.
                    self.last_layer_prior_data = last_layer_prior_data = tf.placeholder(tf.float32,
                                                                                        [None, self.layer_prior_len])
                    code_with_prior = tf.concat([tf.repeat(code, tf.shape(data)[0], axis=0), last_layer_prior_data],
                                                axis=1)
                    code_mode_with_prior = tf.concat(
                        [tf.repeat(tf.expand_dims(posterior.mean(), 0), sample_number, axis=0),
                         last_layer_prior_data, ], axis=1)
                else:
                    code_with_prior = tf.repeat(code, tf.shape(data)[0], axis=0)
                    code_mode_with_prior = tf.repeat(tf.expand_dims(posterior.mean(), 0), sample_number, axis=0)

            with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
                # first layer VAE:
                bernoulli_len = [self.condition_distribution_map_index.shape[0]]
                if bernoulli_len[0] > 0:
                    bernou_likelihood = safe_log(make_bernoulli_decoder(code_with_prior, bernoulli_len).prob(
                        tf.clip_by_value(tensor_indices(data, self.condition_distribution_map_index), 1e-6, 1e7)))
                    if self.merge_dim_likelihood:
                        bernou_likelihood = tf.reduce_sum(bernou_likelihood, axis=-1)
                    bernou_encode_samples = make_bernoulli_decoder(code_mode_with_prior, bernoulli_len).sample()
                    self.bernou_encode_samples = bernou_encode_samples
                    self.to_merge_encode_samples_ops.append(bernou_encode_samples)
                    self.to_merge_encode_samples_idx.append(self.condition_distribution_map_index)
                else:

                    self.bernou_samples = bernou_samples = []
                    self.bernou_encode_samples = bernou_encode_samples = []
                    bernou_likelihood = tf.stop_gradient(tf.get_variable(name='zero_bernou_like',
                                                                         initializer=tf.keras.initializers.zeros,
                                                                         shape=[1, 1]))

                self.to_merge_encode_samples_name.append('bernoulli')
                self.to_merge_encode_samples_likelihood.append(bernou_likelihood)
                if self.weight_loss:
                    self.likelihood = bernou_likelihood * (np.e ** bernoulli_len[0])
                else:
                    self.likelihood = bernou_likelihood

                # second layer VAE:

                def data_condition_not_bound(remove_min_index):
                    cond_index = self.cast_remove_min_index_to_condition_index(remove_min_index)
                    return tf.dtypes.cast(tf.equal(tensor_indices(data, cond_index), 0), tf.float32)

                def sample_condtion_not_bound(remove_min_index):
                    map_index = np.where(np.isin(self.remove_min_index, remove_min_index))[0]
                    return tf.dtypes.cast(tf.equal(tensor_indices(bernou_encode_samples, map_index), 0), tf.float32)

                if len(self.condition_distribution_map_index) == 0 or self.cond_with_gau:
                    code_and_conditional = code_with_prior
                    code_mode_and_conditional = code_mode_with_prior
                else:
                    self.code_and_conditional = code_and_conditional = tf.concat(
                        [code_with_prior, make_cond_embedding(tensor_indices(data, self.condition_distribution_map_index))], axis=1)
                    self.code_mode_and_conditional = code_mode_and_conditional = tf.concat(
                        [code_mode_with_prior, make_cond_embedding(tf.dtypes.cast(bernou_encode_samples, tf.float32))], axis=1)

                def second_layer_vae_construct(index_list, cond_index_list, data_input, decoder, cond_decoder, name):
                    index_len = [index_list.shape[0]]
                    if index_len[0] > 0:
                        distribution, params = decoder(code_and_conditional, index_len, ret_param=True)
                        likelihood = safe_log(distribution.prob(data_input(index_list)))
                        encode_samples = tf.reshape(decoder(code_mode_and_conditional, index_len).sample(), [-1] + index_len)
                        # self.to_merge_encode_samples.append([conti_encode_samples, self.log_norm_index])
                        self.to_merge_encode_samples_ops.append(encode_samples)
                        self.to_merge_encode_samples_idx.append(index_list)
                        if self.merge_dim_likelihood and len(likelihood.shape) > 1:
                            likelihood = tf.reduce_sum(likelihood, axis=-1, keepdims=True)
                    else:
                        encode_samples = None
                        params = None
                        distribution = None
                        likelihood = tf.stop_gradient(
                            tf.get_variable(name='zero_{}_like'.format(name), initializer=tf.keras.initializers.zeros,
                                            shape=[1, 1]))

                    self.to_merge_encode_samples_name.append(name)
                    self.to_merge_encode_samples_likelihood.append(likelihood)
                    if self.weight_loss:
                        self.likelihood += likelihood * (np.e ** index_len[0])
                    else:
                        self.likelihood += likelihood
                    cond_index_len = [cond_index_list.shape[0]]
                    assert cond_index_len[0] == 0
                    encode_samples_cond = None
                    likelihood_cond = tf.stop_gradient(tf.get_variable(name='zero_{}_like_cond'.format(name),
                                                                       initializer=tf.keras.initializers.zeros,
                                                                       shape=[1, 1]))
                    self.to_merge_encode_samples_name.append(name + '-cond')
                    self.to_merge_encode_samples_likelihood.append(likelihood_cond)
                    if self.weight_loss:
                        self.likelihood += likelihood_cond * (np.e ** cond_index_len[0])
                    else:
                        self.likelihood += likelihood_cond
                    return likelihood, encode_samples, likelihood_cond, encode_samples_cond, distribution, params

                # log_norm_distribution
                log_norm_data_input = lambda select_index: tf.clip_by_value(tensor_indices(data, select_index), 1e-6, 1e7)
                self.log_norm_likelihood, self.log_norm_encode_samples, self.log_norm_likelihood_cond, self.log_norm_encode_cond, \
                self.log_norm_distribution, self.log_norm_params = \
                    second_layer_vae_construct(self.log_norm_index, self.log_norm_cond_index, log_norm_data_input,
                                               make_log_norm_decoder, make_log_norm_decoder, 'log_norm')

                discrete_data_input = lambda select_index: tf.dtypes.cast(
                    tf.clip_by_value(tensor_indices(data, select_index), 0, max_category - 1), tf.int32)
                self.disc_likelihood, self.disc_encode_samples, self.disc_likelihood_cond, self.disc_encode_cond, \
                self.discrete_distribution, self.discrete_params = \
                    second_layer_vae_construct(self.discrete_index, self.discrete_cond_index, discrete_data_input,
                                               make_discrete_decoder, make_discrete_decoder, 'discrete')

                norm_data_input = lambda select_index: tensor_indices(data, select_index)
                self.norm_likelihood, self.norm_encode_samples, self.norm_likelihood_cond, self.norm_encode_cond, \
                self.norm_distribution, self.norm_params = \
                    second_layer_vae_construct(self.normal_index, self.normal_cond_index, norm_data_input,
                                               make_normal_decoder, make_normal_decoder, 'norm')

            # likelihood = tf.reduce_mean(conti_likelihood, axis=1) + tf.reduce_mean(disc_likelihood, axis=1) + normal_likelihood + bernou_likelihood  # tf.reduce_mean(bernou_likelihood, axis=1)
            # samples = conti_samples + tf.dtypes.cast(disc_samples, tf.float32) + normal_samples
            # encode_samples = conti_encode_samples + tf.dtypes.cast(disc_encode_samples, tf.float32) + normal_encode_samples
            with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
                self.elbo = elbo = tf.reduce_mean(tf.reduce_mean(self.likelihood) - tf.reduce_mean(divergence))
                if self.optimizer == Optimizer.ADAM:
                    optimize = tf.train.AdamOptimizer(self.lr).minimize(-1 * self.elbo)
                # elif self.optimizer == Optimizer.RADAM:
                #     optimize = RAdamOptimizer(self.lr).minimize(-1 * self.elbo)

        # samples = tf.reshape(make_decoder(prior.sample(10), self.x_shape).mean(), [-1, 28, 28])

        self.data = data
        # self.log_norm_weight = log_norm_weight
        self.sample_number = sample_number
        # self.encode_samples = encode_samples
        self.elbo = elbo
        self.divergence = divergence
        self.code = code
        self.code_mode = code_mode
        self.optimize = optimize

    def encode_sample_merge(self, sess, data, sample_number, prior_data=None):
        # assert isinstance(sess, tf.Session)
        # for item in self.to_merge_encode_samples:
        #     encode_samples = item[0]
        #     indices = item[1]
        if prior_data is not None:
            repeat_times = int(sample_number / prior_data.shape[0])
            cut_sample_number = repeat_times * prior_data.shape[0]
            prior_data_repeat = np.repeat(prior_data, repeat_times, axis=0)
            feed = {self.data: data, self.sample_number: cut_sample_number, self.last_layer_prior_data: prior_data_repeat}

            res_encode_samples = np.zeros(shape=[cut_sample_number] + self.x_shape)
            res_encode_samples[:, self.prior_data_index] += prior_data_repeat
        else:
            feed = {self.data: data, self.sample_number: sample_number}
            res_encode_samples = np.zeros(shape=[sample_number] + self.x_shape)
        encode_samples = sess.run(self.to_merge_encode_samples_ops, feed_dict=feed)
        # res_samples[:, indices] = res_encode_samples
        for idx, samples in zip(self.to_merge_encode_samples_idx, encode_samples):
            res_encode_samples[:, idx] = samples
        return res_encode_samples


class VaeHander(object):
    def __init__(self, vae_list, bandwidth_coef, x_shape, raw_feature_len, downsample, days,
                 shift, scale, init_ops,
                 feature_name, condition_map_features, city_list, merge_learn=False, sess=None, name=''):
        self.vae_list = vae_list
        self.name = name
        self.shift = shift
        self.scale = scale
        self.bandwidth_dict = {}
        self.bandwidth = bandwidth_coef
        self.x_shape = x_shape
        self.raw_feature_len = raw_feature_len
        self.days = days
        self.downsample = downsample
        self.city_list = city_list
        self.condition_map_features = condition_map_features
        self.feature_name = feature_name
        self.merge_learn = merge_learn
        self.kde_map = {}
        self.sess = sess
        for vae in self.vae_list[1:]:
            assert vae.reuse_encoder, "all layer of vae should use the same z embdding, except first layer."
        self.daily_data_distribution_ph = self.vae_list[0].data
        self.shu_idx = None
        self.have_kde = False
        if len(self.vae_list) == 1:
            self.relation_day_sampling_ratio = 1/15
        else:
            self.relation_day_sampling_ratio = 1/3
        self.relation_day_size = int(self.days * self.relation_day_sampling_ratio) + 1
        self.relation_matrix = np.zeros([self.city_size, self.city_size, self.relation_day_size, self.relation_day_size])
        relation_matrix = self.load_relation_matrix()
        if relation_matrix is not None and relation_matrix.shape == self.relation_matrix.shape:
            self.relation_matrix = relation_matrix
            logger.info("[load relation matrix succeed]")
            self.init_relation_matrix = True
        else:
            self.init_relation_matrix = False
        self.load_kde_map()
        if init_ops:
            if self.merge_learn:
                self.init_train()
            self._init_embedding_predict_network()

    @property
    def city_size(self):
        return len(self.city_list)

    @property
    def trainable_variable(self):
        return tf.trainable_variables(self.name)

    def load_relation_matrix(self):
        return tester.load_pickle('relation_matrix-c_{}-d_{}-{}'.format(self.city_size, self.relation_day_size, self.x_shape))

    def save_relation_matrix(self):
        if not self.init_relation_matrix:
            tester.save_pickle(self.relation_matrix, 'relation_matrix-c_{}-d_{}-{}'.format(self.city_size, self.relation_day_size, self.x_shape))

    def load_from_checkpoint(self, checkpoint_path, target_prefix_name=None):
        import os.path as osp
        logger.info("ckpt path  {}".format(checkpoint_path))
        ckpt_path = tf.train.latest_checkpoint(checkpoint_path)
        if target_prefix_name is None:
            target_prefix_name = self.name + '/'
        var_list = self.trainable_variable
        var_dict = {}
        for v in var_list:
            key_name = v.name[v.name.find(self.name) + len(self.name) + 1:]
            # key_name = '/'.join(v.name.split('/')[1:])
            key_name = key_name.split(':')[0]
            var_dict[target_prefix_name + key_name] = v
        self.sess.run(tf.initialize_variables(var_list))
        saver = tf.train.Saver(var_list=var_dict)
        # U.initialize()
        logger.info("ckpt_path [load vae handle] {}".format(ckpt_path))
        saver.restore(self.sess, ckpt_path)
        print("----------load {}simulator model- successfully---------".format(self.name))
        return int(ckpt_path.split('-')[-1])

    @property
    def z_size(self):
        return sum([vae.z_size for vae in self.vae_list if not vae.reuse_encoder])

    @property
    def remove_min_index(self):
        res = []
        for vae in self.vae_list:
            res = np.append(res, vae.remove_min_index)
        return np.unique(res).astype('int32')

    @property
    def condition_index_bound_val(self):
        return self.vae_list[0].condition_index_bound_val

    def infer_z_op(self):
        return self.vae_list[0].code_mode

    def z_step(self, obs):
        return self.sess.run(self.infer_z_op(), feed_dict={self.daily_data_distribution_ph: obs})

    def filter_norm_obs(self, original_obs, original_feature_name):
        select_index = get_index_from_names(original_feature_name, self.feature_name)
        filter_obs = original_obs[:, select_index]
        assert len(select_index) == self.raw_feature_len
        # norm_obs = (filter_obs - self.shift[:self.raw_feature_len]) / self.scale[:self.raw_feature_len]
        condition_index_bound_val = self.condition_index_bound_val
        remove_min_index = self.remove_min_index
        filter_obs[:, remove_min_index] = np.round(filter_obs[:, remove_min_index], 2)
        min_features_val = condition_index_bound_val[remove_min_index]
        remove_min_features = filter_obs[:, remove_min_index]
        is_min_features = min_features_val == remove_min_features
        extend_features = np.concatenate([filter_obs, is_min_features.astype('float32')], axis=-1)
        norm_obs = (extend_features - self.shift) / self.scale
        return norm_obs

    def _init_embedding_predict_network(self):
        with tf.variable_scope('evaluation', reuse=tf.AUTO_REUSE):
            x = self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.z_size * 2], name='input_data')
            for _ in range(1):
                x = tf.layers.dense(x, 32, tf.nn.tanh)
            logits = tf.layers.dense(x, 1)
            self.predict = logits # tf.nn.sigmoid(logits)
            y = self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='label')
            self.loss = tf.losses.mean_squared_error(labels=y, predictions=logits) # tf.losses.sigmoid_cross_entropy(y, logits, label_smoothing=0.1)
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.opt = optimizer.minimize(self.loss)
            self.evaluation_param = tf.trainable_variables('evaluation') + optimizer.variables()
            self.init_param = tf.initialize_variables(self.evaluation_param)

    def embedding_predictable_evaluation(self, city_data_dict):
        # calculate relation matrix
        def eval_kl(city_a, city_b, day):
            sample_data = city_data_dict[city_a][0][int(day / self.relation_day_sampling_ratio)] # [:, self.vae_list[1].layer_filter_index]
            kde_a = self.kde_map[city_a + '/overall']
            kde_b = self.kde_map[city_b + '/overall']
            shuffle_index = np.random.randint(0, sample_data.shape[0], min(1000, sample_data.shape[0]))
            score_a = kde_a.score_samples(sample_data[shuffle_index])
            score_b = kde_b.score_samples(sample_data[shuffle_index])
            kl = np.mean(np.clip(score_a - score_b, None, 20))
            return kl

        self.z_matrix = np.zeros([self.city_size, self.city_size, self.relation_day_size, self.relation_day_size, self.z_size * 2])
        for ida in range(self.city_size):
            for idb in range(ida, self.city_size):
                for day_a in range(0, self.relation_day_size):
                    for day_b in range(day_a, self.relation_day_size):
                        city_a, city_b = self.city_list[ida], self.city_list[idb]
                        logger.info("[calculate relation between {} and {} on day {}-{}]".format(city_a, city_b, day_a / self.relation_day_sampling_ratio, day_b / self.relation_day_sampling_ratio))
                        z_a = self.z_step(city_data_dict[city_a][0][int(day_a / self.relation_day_sampling_ratio)])
                        z_b = self.z_step(city_data_dict[city_b][0][int(day_b / self.relation_day_sampling_ratio)])
                        self.z_matrix[ida, idb, day_a, day_b, :] = self.z_matrix[ida, idb, day_b, day_a, :] \
                            = self.z_matrix[idb, ida, day_a, day_b, :] = self.z_matrix[idb, ida, day_b, day_a, :] \
                            = np.concatenate([z_a, z_b], axis=0)
                        if not self.init_relation_matrix:
                            kl = eval_kl(city_a, city_b, day_a)
                            self.relation_matrix[ida, idb, day_a, day_b] = self.relation_matrix[ida, idb, day_b, day_a] = \
                                self.relation_matrix[idb, ida, day_a, day_b] = self.relation_matrix[idb, ida, day_b, day_a] = kl
        if not self.init_relation_matrix:
            self.save_relation_matrix()
            self.init_relation_matrix = True
        # construct dataset.
        x = np.reshape(self.z_matrix, [-1, self.z_size * 2])
        y = np.reshape(self.relation_matrix, [-1, 1])
        # y_median = np.median(y)
        y_label = y # (y > y_median).astype('int16')
        if self.shu_idx is None:
            self.shu_idx = np.arange(x.shape[0])
        np.random.shuffle(self.shu_idx)
        x = x[self.shu_idx]
        y_label = y_label[self.shu_idx]
        self.sess.run(self.init_param)
        train_x = x[:int(x.shape[0] * 0.7)]
        train_y = y_label[:int(x.shape[0] * 0.7)]
        original_ts = tester.time_step_holder.get_time()
        min_error = 100
        for i in range(2048):
            loss, _ = self.sess.run([self.loss, self.opt], feed_dict={self.x: train_x, self.y: train_y})
            if i % 50 == 0:
                logger.info("({}) loss {}".format(i, loss))
            if loss < 0.00001:
                logger.info("({}) loss {}".format(i, loss))
                break
            if i % 200 == 0:
                test_x = x[int(x.shape[0] * 0.7):]
                test_y = y_label[int(x.shape[0] * 0.7):]
                predict = self.sess.run(self.predict, feed_dict={self.x: test_x})
                precision = np.mean(np.abs(predict - test_y))
                min_error = min(precision, min_error)
                # logger.record_tabular("performance/mae", precision)
                logger.dumpkvs()
                # tester.time_step_holder.inc_time()

        test_x = x[int(x.shape[0] * 0.7):]
        test_y = y_label[int(x.shape[0] * 0.7):]
        predict = self.sess.run(self.predict, feed_dict={self.x: test_x})
        precision = np.mean(np.abs(predict - test_y))
        min_error = min(precision, min_error)
        logger.record_tabular("performance/mae", min_error)
        real_error_std = np.mean(np.std(self.relation_matrix, axis=(2, 3)))
        logger.record_tabular("performance/real_error_std", real_error_std)
        np.std(self.relation_matrix, axis=(0, 1))
        tester.time_step_holder.set_time(original_ts)
        logger.dumpkvs()

        return min_error

    def opt_bandwidth(self, input_data, test_city, name, coef=1):
        if self.have_kde:
            return self.bandwidth_dict[name]
        else:
            def opt_unit(data):
                shuffle_index = np.random.randint(0, input_data.shape[0], min(4000, input_data.shape[0]))
                params = {'bandwidth': np.logspace(-1, 3, 20)}
                grid = GridSearchCV(KernelDensity(rtol=5e-2), params, cv=5, iid=False)
                grid.fit(data[shuffle_index])
                return grid

            # if test_city not in self.bandwidth_dict:
            #     self.bandwidth_dict[test_city] = {}
            # use grid search cross-validation to optimize the bandwidth
            if name not in self.bandwidth_dict:
                grid = tester.time_used_wrap("opt_bandwidth-" + name, opt_unit, input_data)
                bandwidth = grid.best_params_['bandwidth'] * self.bandwidth * coef
                self.bandwidth_dict[name] = bandwidth
            else:
                bandwidth = self.bandwidth_dict[name]
            return bandwidth

    def save_kde_map(self):
        if not self.have_kde:
            tester.save_pickle(self.kde_map, 'kde-bandwidth_ecoff-{}-c:{}-{}'.format(self.bandwidth, self.city_size, self.x_shape))
            tester.save_pickle(self.bandwidth_dict, 'kde-bandwidth_ecoff_dict-{}-c:{}-{}'.format(self.bandwidth, self.city_size, self.x_shape))
            self.have_kde = True

    def load_kde_map(self):
        self.kde_map = tester.load_pickle('kde-bandwidth_ecoff-{}-c:{}-{}'.format(self.bandwidth, self.city_size, self.x_shape))
        self.bandwidth_dict = tester.load_pickle('kde-bandwidth_ecoff_dict-{}-c:{}-{}'.format(self.bandwidth, self.city_size, self.x_shape))
        if self.kde_map is None or self.bandwidth_dict is None:
            logger.info("[load kde failed]")
            self.have_kde = False
            self.kde_map = {}
            self.bandwidth_dict = {}
        else:
            logger.info("[load kde succeed]")
            self.have_kde = True

    def compute_kl(self, real_data, fake_data, bandwidth, kde_name):
        if fake_data.shape[0] == 0:
            return None, None, None
        shuffle_index = np.random.randint(0, real_data.shape[0], min(2000, real_data.shape[0]))
        fake_shuffle_index = np.random.randint(0, fake_data.shape[0], min(2000, fake_data.shape[0]))
        if kde_name not in self.kde_map:
            kde_real = KernelDensity(bandwidth=bandwidth, rtol=5e-2).fit(real_data[shuffle_index])
            self.kde_map[kde_name] = kde_real
        kde_real = self.kde_map[kde_name]
        kde_fake = KernelDensity(bandwidth=bandwidth, rtol=5e-2).fit(fake_data[fake_shuffle_index])
        score_real = kde_real.score_samples(real_data[shuffle_index])
        score_fake = kde_fake.score_samples(real_data[shuffle_index])
        kl = np.mean(np.clip(score_real - score_fake, None, 20))
        return kl, kde_real, kde_fake

    @staticmethod
    def log_func(epoch, prefix_file_name, test_city, vae_name, detail_kv, summary_kv):
        tester.time_step_holder.set_time(epoch)
        for k, v in detail_kv.items():
            logger.record_tabular("({}){}-{}/{}".format(prefix_file_name, test_city, vae_name, k), np.mean(v))
        for k, v in summary_kv.items():
            logger.record_tabular("{}/{}-({}){}".format(k, vae_name, prefix_file_name, test_city), np.mean(v))
            # logger.record_tabular("({}){}-{}/{}".format(prefix_file_name, test_city, vae_name, k), v)
        logger.record_tabular("info/epoch", epoch)
        logger.dump_tabular()

    def init_train(self):

        with tf.variable_scope('merge_loss', reuse=tf.AUTO_REUSE):

            elbos = 0
            kld = 0
            likelihood = 0
            lr = 0
            for vae in self.vae_list:
                likelihood += vae.likelihood
                lr = vae.lr
                kld = vae.divergence

            elbos = tf.reduce_mean(likelihood) - tf.reduce_mean(kld)
            self.optimize = tf.train.AdamOptimizer(lr)
            grads_and_vars = self.optimize.compute_gradients(-1 * elbos, tf.trainable_variables('distribution_emb'))
            self.training_op = self.optimize.apply_gradients(grads_and_vars)
        for grad, var in grads_and_vars:
            if 'LayerNorm' in var.name or 'bias' in var.name:
                continue
            if grad is not None:
                tf.summary.histogram(str(var.name).replace('/', '-') + '/gradient', grad)
            tf.summary.histogram(str(var.name).replace('/', '-') + '/weight', var)

        self.summary = tf.summary.merge_all()

    def train(self, next_batch_data):
        feed = {}
        opts = []
        for vae in self.vae_list:
            assert isinstance(vae, VAE)
            if vae.layer_prior_len == 0:
                feed[vae.data] = next_batch_data
            else:

                feed[vae.data] = next_batch_data
                feed[vae.last_layer_prior_data] = next_batch_data[:, vae.prior_data_index]
            opts.append(vae.optimize)
        if self.merge_learn:
            _, summary = tester.run_with_metadata(self.sess, [self.training_op, self.summary], feed, 'train')
            tester.add_summary(summary)
        else:
            tester.run_with_metadata(self.sess, opts, feed, 'train')

    def evaluation(self, real_data_info, test_city, prefix_file_name, epoch, plot_pic):
        logger.info("start evaluation")
        oracle_data_sample_number = 4000
        real_data = real_data_info[0]
        shuffle_index = np.random.randint(0, real_data.shape[1], min(oracle_data_sample_number, real_data.shape[1]))
        real_data = real_data[:, shuffle_index]
        real_data_mean = real_data_info[1]
        real_data_std = real_data_info[2]
        real_data_min = real_data_info[3]
        real_data_max = real_data_info[4]
        assert np.all(self.shift == real_data_mean) and np.all(self.scale == real_data_std)

        def infer_data_unit(vae, input_data, prior_data_real, prior_data_sample):
            assert isinstance(vae, VAE)
            if prior_data_real is not None:
                feed = {vae.data: input_data, vae.last_layer_prior_data: prior_data_real}
            else:
                feed = {vae.data: input_data}
            ops = [vae.elbo, vae.divergence, vae.likelihood, vae.to_merge_encode_samples_likelihood, vae.code_mode]
            test_elbo, test_divergence, test_likelihood, test_likelihood_part, test_codes = \
                tester.run_with_metadata(self.sess, ops, feed, vae.name + '-evaluation')

            def get_valid_sample_index(samples):
                rescale_sample = samples * real_data_std + real_data_mean
                valid_sample_index = np.where(np.logical_and(np.all(rescale_sample >= real_data_min, axis=-1),
                                                             np.all(rescale_sample <= real_data_max, axis=-1)))
                return valid_sample_index

            test_samples = vae.encode_sample_merge(self.sess, input_data, oracle_data_sample_number * 10, prior_data_sample)
            test_samples[:, vae.condition_distribution_map_index] = np.clip(
                np.round(test_samples[:, vae.condition_distribution_map_index]), 0, 1)
            valid_sample_index = get_valid_sample_index(test_samples)
            logger.record_tabular("samples/valid_ratio_overall", float(valid_sample_index[0].shape[0]) / (oracle_data_sample_number * 10))
            test_samples = test_samples[valid_sample_index]
            logger.info("start compute kde")
            bandwidth = self.opt_bandwidth(input_data, test_city, vae.name, 2)
            detail_kv = {"elbo": test_elbo, "divergence": test_divergence, "likelihood": test_likelihood, }
            summary_kv = {"elbo": test_elbo, "divergence": test_divergence, "likelihood": test_likelihood}
            if test_samples.shape[0] > 0:
                kl, _, _ = tester.time_used_wrap("compute-kl-" + vae.name, self.compute_kl,
                                                 input_data[:, vae.layer_filter_index],
                                                 test_samples[:, vae.layer_filter_index], bandwidth, test_city + '/' + vae.name)
                if kl is not None:
                    detail_kv['kl'] = kl
                    summary_kv['kl'] = kl
            else:
                detail_kv['kl'] = 20
                summary_kv['kl'] = 20
            for likelihood_part, name in zip(test_likelihood_part, vae.to_merge_encode_samples_name):
                detail_kv[name] = np.mean(likelihood_part)
            self.log_func(epoch, prefix_file_name, test_city, vae.name, detail_kv, summary_kv)
            if prior_data_real is not None:
                shuffle_index = np.random.randint(0, prior_data_real.shape[0], oracle_data_sample_number)
                test_samples_single_layer = vae.encode_sample_merge(self.sess, input_data, oracle_data_sample_number * 10,
                                                                    prior_data_real[shuffle_index])
            else:
                test_samples_single_layer = vae.encode_sample_merge(self.sess, input_data, oracle_data_sample_number * 10, None)

            test_samples_single_layer[:, vae.condition_distribution_map_index] = np.clip(
                np.round(test_samples_single_layer[:, vae.condition_distribution_map_index]), 0, 1)
            valid_sample_index = get_valid_sample_index(test_samples_single_layer)
            test_samples_single_layer = test_samples_single_layer[valid_sample_index]
            logger.record_tabular("samples/valid_ratio_{}".format(vae.name), float(valid_sample_index[0].shape[0]) / (oracle_data_sample_number * 10))

            return summary_kv, test_samples, test_codes, test_samples_single_layer

        static_col_limit = 6
        rows = int(max(int(self.days / self.downsample), self.vae_list[0].x_shape[0] / static_col_limit))
        cols = int(static_col_limit + sum([x.layer_vae_feature_len for x in self.vae_list[1:]]))
        if plot_pic:
            fig, ax = plt.subplots(nrows=rows + 1, ncols=cols, figsize=(6 * cols, 6 * rows))

        def plot_pic_func(vae, start_cols, start_rows, col_limit, plot_samples, plot_samples_single_layer, day):
            index = 0
            plot_count = 0
            plot_real_std = real_data_std
            plot_real_mean = real_data_mean
            plot_real = real_data[day]
            assert plot_samples.shape[-1] == plot_real_std.shape[-1]
            assert plot_real.shape[-1] == plot_real_std.shape[-1]
            if plot_samples_single_layer.shape[0] == 0 or plot_samples.shape[0] == 0:
                return
            for test_f, test_single_f, real_f in zip(plot_samples.T, plot_samples_single_layer.T, plot_real.T):
                # index should be in this layer vae
                if index not in vae.layer_filter_index:
                    index += 1
                    continue
                cur_static_feature_name = self.feature_name[index].split('.')[-1]
                if self.feature_name[index] in self.condition_map_features:
                    c_i = np.where(self.condition_map_features == self.feature_name[index])[0]
                    assert c_i.shape[0] == 1
                    c_i = c_i[0]
                    test_f = test_f[np.where(np.clip(np.round(plot_samples[:, c_i + self.raw_feature_len]), 0, 1) == 0)]
                    test_single_f = test_single_f[
                        np.where(np.clip(np.round(plot_samples_single_layer[:, c_i + self.raw_feature_len]), 0, 1) == 0)]
                    real_f = real_f[np.where(plot_real[:, c_i + self.raw_feature_len] == 0)]
                    if real_f.shape[0] == 0:
                        plot_count += 1
                        index += 1
                        continue
                # HIST
                data = [test_f * plot_real_std[index] + plot_real_mean[index],
                        real_f * plot_real_std[index] + plot_real_mean[index],
                        test_single_f * plot_real_std[index] + plot_real_mean[index], ]

                def data_clip(to_clip_data):
                    # assert np.where(to_clip_data > (data[1].max() + 1) * 1.5)[0].shape[0] == 0
                    # assert np.where(to_clip_data < (data[1].min()) * 0.5 - 1)[0].shape[0] == 0
                    to_clip_data[np.where(to_clip_data > (data[1].max() + 1) * 1.5)] = (data[1].max() + 1) * 1.5
                    to_clip_data[np.where(to_clip_data < (data[1].min()) * 0.5 - 1)] = (data[1].min()) * 0.5 - 1
                    return to_clip_data

                data_min_list = []
                data_max_list = []
                def safe_operator(data_input):
                    if data_input.shape[0] != 0:
                        # data_input = data_clip(data_input)
                        data_min_list.append(data_input.min())
                        data_max_list.append(data_input.max())
                    return data_input
                data[0] = safe_operator(data[0])
                data[1] = safe_operator(data[1])
                data[2] = safe_operator(data[2])
                data_min = np.min(data_min_list)
                data_max = np.min(data_max_list)

                select_ax = ax[start_rows + int(plot_count / col_limit), start_cols + plot_count % col_limit]
                single_bandwidth = self.opt_bandwidth(data[1].reshape(-1, 1), test_city, cur_static_feature_name, 2)
                if test_f.shape[0] != 0 and test_single_f.shape[0] != 0:
                    kl_single, kde_real_single, kde_fake_single = self.compute_kl(data[1].reshape(-1, 1),
                                                                                  data[0].reshape(-1, 1),
                                                                                  single_bandwidth, test_city + '/' + cur_static_feature_name)
                else:
                    kl_single = 20
                    kde_real_single = kde_fake_single = None
                # kde_fake_single = KernelDensity(bandwidth=single_bandwidth).fit(data[0].reshape(-1, 1))
                # kde_real_single = KernelDensity(bandwidth=single_bandwidth).fit(data[1].reshape(-1, 1))
                xs = np.linspace(data[1].min(), data[1].max(), 100).reshape(-1, 1)
                if kde_real_single is not None:
                    if test_f.shape[0] != 0 and test_single_f.shape[0] != 0:
                        fake_prob = np.exp(kde_fake_single.score_samples(xs))
                    real_prob = np.exp(kde_real_single.score_samples(xs))
                # kl_single = np.mean(np.clip(
                #     kde_real_single.score_samples(data[1].reshape(-1, 1)) - kde_fake_single.score_samples(
                #         data[1].reshape(-1, 1)), None, 20))

                    x_f = select_ax.hist(data, bins=30, density=True, histtype='step',
                                         label=['fake', 'real', 'fake-single'],
                                         range=(data_min, data_max))
                    if test_f.shape[0] != 0 and test_single_f.shape[0] != 0:
                        x_f = select_ax.plot(xs, fake_prob, label='fake-kde')
                    x_r = select_ax.plot(xs, real_prob, label='real-kde')
                # x_r = ax[int(index / 4), index % 4].hist(, bins=20, density=True, alpha=0.9,histtype='step', label='real')
                # ax[int(index / 4), index % 4].legend((x_f[0], x_r[0]), ('fake', 'real'))
                select_ax.legend(prop={'size': 10})
                select_ax.set_title(
                    "{}-{}-{:.3f}-{:.3f}".format(test_city, cur_static_feature_name, kl_single, single_bandwidth))

                index += 1
                plot_count += 1

        vae = self.vae_list[0]
        logger.info("infer static data.")
        summary_kv_merge, test_samples_static, static_test_codes, test_samples_single_layer_static = infer_data_unit(
            vae, real_data[0], None, None)

        if plot_pic:
            plot_pic_func(vae, 0, 0, static_col_limit, test_samples_static, test_samples_single_layer_static, day=0)

        def dict_merge(new_dict):
            for k, v in summary_kv_merge.items():
                if not isinstance(summary_kv_merge[k], list):
                    summary_kv_merge[k] = np.array([summary_kv_merge[k]])
                if k in new_dict:
                    summary_kv_merge[k] = np.append(summary_kv_merge[k], new_dict[k])

        test_codes_merge = []
        for d in range(0, self.days, self.downsample):
            # test_samples_single_day = test_samples_static.copy()
            # test_samples_single_layer_single_day = test_samples_single_layer_static.copy()
            # multi_test_codes = static_test_codes.copy()
            input_data = real_data[d]
            summary_kv, test_samples_single_day, multi_test_codes, test_samples_single_layer_single_day = infer_data_unit(
                self.vae_list[0], input_data, None, None)
            del summary_kv['kl']
            dict_merge(summary_kv)
            start_cols = static_col_limit
            for vae in self.vae_list[1:]:
                assert isinstance(vae, VAE)
                if test_samples_single_day.shape[0] == 0 or test_samples_single_layer_single_day.shape[0] == 0:
                    break
                input_data = real_data[d]
                prior_data_sample = test_samples_single_day[:, vae.prior_data_index]
                prior_data_real = input_data[:, vae.prior_data_index]
                summary_kv, test_samples, test_codes, test_samples_single_layer = infer_data_unit(vae, input_data,
                                                                                                  prior_data_real,
                                                                                                  prior_data_sample)
                del summary_kv['kl']
                dict_merge(summary_kv)
                # assert all later-layer features are added to where original value is zero.
                # assert test_samples_single_day.sum(axis=0)[np.where(test_samples.sum(axis=0) > 0)].sum() == 0
                test_samples_single_day = test_samples
                test_samples_single_layer_single_day = test_samples_single_layer
                if not vae.reuse_encoder:
                    multi_test_codes = np.append(multi_test_codes, test_codes)
                else:
                    # assert np.any(multi_test_codes==test_codes)
                    multi_test_codes = test_codes
                if plot_pic:
                    col_limit = len(vae.layer_filter_index)
                    plot_pic_func(vae, start_cols, int(d / self.downsample), col_limit, test_samples_single_day,
                                  test_samples_single_layer_single_day, day=d)
                    start_cols += len(vae.layer_filter_index)

            bandwidth = self.opt_bandwidth(input_data, test_city, 'overall', 2)
            kl, _, _ = self.compute_kl(input_data, test_samples_single_day, bandwidth, test_city + '/overall')
            # kde_real = KernelDensity(bandwidth=bandwidth).fit(input_data)
            # kde_fake = KernelDensity(bandwidth=bandwidth).fit(test_samples_single_day)
            # score_real = kde_real.score_samples(input_data)
            # score_fake = kde_fake.score_samples(input_data)
            # kl = np.mean(np.clip(score_real - score_fake, None, 20))
            if kl is not None:
                dict_merge({"kl": kl})
            test_codes_merge.append(multi_test_codes)
        self.log_func(epoch, prefix_file_name, test_city, 'overall', {}, summary_kv_merge)
        if plot_pic:
            os.makedirs("{}/{}/".format(tester.results_dir, prefix_file_name), exist_ok=True)
            plt.savefig('{}/{}/{}-vae-mnist-{}.png'.format(tester.results_dir, prefix_file_name, test_city, epoch),
                        dpi=80)
        return test_codes_merge


if __name__ == '__main__':
    # test: 4, 2, 10
    vae = VAE(hid_size=200, hid_layers=2, lr=0.001, x_shape=[28, 28], z_size=2, decode_type=VAE.BERNOULLI)
    vae.construction()
    vae.train_demo()
