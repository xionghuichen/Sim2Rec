import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
from common import logger
from stable_baselines.common import tf_util

class VAE(object):

    def __init__(self, hid_size, hid_layers, layer_norm, constant_z_scale, lr,
                 res_struc, x_shape, z_size, gau_index, sess, name, l2_reg_coeff):
        self.hid_size = hid_size
        self.hid_layers = hid_layers
        self.res_struc = res_struc
        self.layer_norm = layer_norm
        self.constant_z_scale = constant_z_scale
        self.x_shape = x_shape
        self.z_size = z_size
        self.name = name
        self.gau_index = gau_index
        self.lr = lr
        self.l2_reg_coeff = l2_reg_coeff
        self.sess = sess
        self._init()
        pass

    def _init(self):
        def make_encoder(data, code_size, name='encoder'):
            with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
                # calculate multi gaussian: http://www.tina-vision.net/docs/memos/2003-003.pdf
                # https://github.com/katerakelly/oyster/blob/cd09c1ae0e69537ca83004ca569574ea80cf3b9c/rlkit/torch/sac/agent.py#L10
                with tf.compat.v1.variable_scope('emb'):
                    x = tf.keras.layers.Flatten()(data)
                    o_x = x = tf.keras.layers.Dense(self.hid_size, tf.nn.leaky_relu)(x)
                    for _ in range(self.hid_layers - 1):
                        x = tf.keras.layers.Dense(self.hid_size)(x)
                        if self.layer_norm:
                            x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
                        x = tf.nn.leaky_relu(x)
                    if self.res_struc:
                        x = o_x + x
                locs = tf.keras.layers.Dense(code_size, name='loc')(x)
                if self.constant_z_scale:
                    scales = locs * 0 + 1
                else:
                    scales = tf.keras.layers.Dense(code_size, tf.nn.softplus, name='scale')(x)
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


        def make_decoder(code, data_shape, name='', ret_param=False):
            with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
                data_size = np.prod(data_shape)
                # x = code
                with tf.compat.v1.variable_scope('emb'):
                    x = code
                    o_x = x = tf.keras.layers.Dense(self.hid_size, tf.nn.leaky_relu)(x)
                    for _ in range(self.hid_layers - 1):
                        x = tf.keras.layers.Dense(self.hid_size)(x)
                        if self.layer_norm:
                            x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
                        x = tf.nn.leaky_relu(x)
                    if self.res_struc:
                        x = x + o_x

                    loc = tf.keras.layers.Dense(data_size, name='loc')(x)
                    scale = tf.keras.layers.Dense(data_size, tf.nn.softplus, name='scale')(x)
                    loc = tf.reshape(loc, [-1] + data_shape)
                    scale = tf.reshape(scale, [-1] + data_shape)
                    distribution = tfd.MultivariateNormalDiag(loc, scale,
                                                      allow_nan_stats=False)
                    dis_param = (loc, scale)  # tfd.MultivariateNormalDiag(loc, scale, allow_nan_stats=False) # tfd.Independent(, len(data_shape)-1)
                if ret_param:
                    return distribution, dis_param
                else:
                    return distribution

        def tensor_indices(data, indices):
            return tf.transpose(
                tf.gather_nd(tf.transpose(data), indices=np.expand_dims(indices, axis=1), batch_dims=0, ))
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            def safe_log(ops):
                return tf.math.log(ops + 1e-6)
            sample_number = tf.compat.v1.placeholder(tf.int32, [])
            make_gau_decoder = lambda code, data_shape, name='', ret_param=False: make_decoder(code, data_shape, name=name, ret_param=ret_param)
            with tf.compat.v1.variable_scope('encode', reuse=tf.compat.v1.AUTO_REUSE):
                # Define the model.
                data = tf.compat.v1.placeholder(tf.float32, [None, ] + self.x_shape, name='data_ph')
                prior = make_prior(code_size=self.z_size)
                posterior = make_encoder(data, code_size=self.z_size)
                code = posterior.sample()
                code_mode = posterior.mode()
                self.reuse_encoder = False
                # Define the loss.MultivariateNormalDiag
                code = tf.expand_dims(code, 0)  # TODO: don't use expand_dims
                divergence = tfd.kl_divergence(posterior, prior)
                code_with_prior = tf.repeat(code, tf.shape(data)[0], axis=0)
                code_mode_with_prior = tf.repeat(tf.expand_dims(posterior.mean(), 0), sample_number, axis=0)
            with tf.compat.v1.variable_scope('decode', reuse=tf.compat.v1.AUTO_REUSE):
                code_and_conditional = code_with_prior
                code_mode_and_conditional = code_mode_with_prior

                data_input = data
                index_list = self.gau_index
                index_len = [index_list.shape[0]]
                distribution, params = make_gau_decoder(code_and_conditional, index_len, ret_param=True)
                likelihood = safe_log(distribution.prob(data_input))
                self.encode_samples = tf.reshape(make_gau_decoder(code_mode_and_conditional, index_len).sample(), [-1] + index_len)
                self.likelihood = tf.reduce_sum(likelihood, axis=-1, keepdims=True)
            with tf.compat.v1.variable_scope('loss', reuse=tf.compat.v1.AUTO_REUSE):
                self.elbo = elbo = tf.reduce_mean(tf.reduce_mean(self.likelihood) - tf.reduce_mean(divergence))
                l2_vars = tf_util.get_trainable_vars(self.name)
                self.l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in l2_vars]) * self.l2_reg_coeff
                optimize = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(-1 * self.elbo + self.l2_loss)
            self.data = data
            self.divergence = divergence
            self.sample_number = sample_number
            self.code = code
            self.code_mode = code_mode
            self.optimize = optimize

    @property
    def trainable_variable(self):
        return tf.trainable_variables(self.name)

    def reconstruct_samples(self, data, sample_number):
        feed = {self.data: data, self.sample_number: sample_number}
        # res_encode_samples = np.zeros(shape=[sample_number] + self.x_shape)
        code, reconstructed_samples = self.sess.run([self.code, self.encode_samples], feed_dict=feed)
        return code, reconstructed_samples

    def embedding(self, data, deter=False):
        feed = {self.data: data}
        # res_encode_samples = np.zeros(shape=[sample_number] + self.x_shape)
        if deter:
            code = self.sess.run([self.code_mode], feed_dict=feed)
        else:
            code = self.sess.run([self.code], feed_dict=feed)[0]
        return code

    def train(self, next_batch_data):
        feed = {}
        opts = []
        feed[self.data] = next_batch_data
        opts.append(self.optimize)
        opts.extend([self.elbo, self.likelihood, self.divergence, self.l2_loss, self.code])
        return self.sess.run(opts, feed_dict=feed)[1:]

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
