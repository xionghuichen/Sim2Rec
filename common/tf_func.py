import tensorflow as tf


def noam_scheme(global_step, num_warmup_steps, num_train_steps, init_lr, end_init_lr, warmup=True):
    """
    decay learning rate
    if warmup > global step, the learning rate will be global_step/num_warmup_steps * init_lr
    if warmup < global step, the learning rate will be polynomial decay
    :param global_step: global steps
    :param num_warmup_steps: number of warm up steps
    :param num_train_steps: number of train steps
    :param init_lr: initial learning rate
    :param end_lr: final learning rate
    :param warmup: if True, it will warm up learning rate
    :return: learning rate
    """
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(learning_rate,
                                                global_step,
                                                num_train_steps,
                                                end_learning_rate=0.0,
                                                power=1.0,
                                                cycle=False)

    if warmup:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate


def normalize(tensor, stats, just_scale):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
    if stats is None:
        return tensor
    if just_scale:
        return tensor / stats.std

    else:
        return (tensor - stats.mean) / stats.std


def normalize_np(tensor, stats, sess, just_scale):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """

    if stats is None:
        return tensor
    std, mean = sess.run([stats.std, stats.mean])
    if just_scale:
        return tensor / std
    else:
        return (tensor - mean) / std

def denormalize_np(tensor, stats, sess, just_scale):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return tensor
    std, mean = sess.run([stats.std, stats.mean])
    if just_scale:
        return tensor * std
    else:
        return tensor * std + mean