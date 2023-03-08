import argparse
import sys

sys.path.append('../../')
import tensorflow as tf
import os

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from common import logger
from common.utils import *
from common.config import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from vae.src.private_config import *
from common.tester import Tester, tester
from vae.src.vae_cls_fix import VAE, VaeHander
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

from common.env_base import BaseDriverEnv

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DEMER")
    # task parameters
    parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--task', help='environment ID', default='hier_ae')
    # parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--vae_task_type', help='environment ID', default=VaeTaskType.STATIC_COUPON)
    parser.add_argument('--vae_data_type', help='environment ID', default=VaeDataType.NORM)
    # data load setting
    parser.add_argument('--start_date', type=int, default=20190601)
    parser.add_argument('--end_date', type=int, default=20190630)
    parser.add_argument('--city', type=str, default='Weifang')
    parser.add_argument('--expert_path', type=str, default=DATA_SET_ROOT)
    parser.add_argument('--folder_name', type=str, default='_ceil_reduce_single_day_coupon_29')
    parser.add_argument('--evaluate_percent', type=int, default=1)
    parser.add_argument('--bandwidth', type=int, default=5)

    # vae parameters
    parser.add_argument('--z_size', type=int, default=50)
    boolean_flag(parser, 'share_codes', default=True)
    boolean_flag(parser, 'merge_dim_likelihood', default=True)
    boolean_flag(parser, 'merge_learn', default=True)
    boolean_flag(parser, 'split_z', default=False)
    boolean_flag(parser, 'split_z_by_layer', default=False)
    parser.add_argument('--hid_layers', type=int, default=2)
    parser.add_argument('--hid_size', type=int, default=512)
    parser.add_argument('--scale_hid', type=int, default=1)
    parser.add_argument('--cond_embedding_layer', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    boolean_flag(parser, 'split_condition_predict', default=False)  # 丢弃参数
    boolean_flag(parser, 'layer_norm', default=True)
    boolean_flag(parser, 'weight_loss', default=False)  # 负面作用
    boolean_flag(parser, 'zero_init', default=True)
    boolean_flag(parser, 'constant_z_scale', default=False)
    boolean_flag(parser, 'cond_with_gau', default=False)
    # train parameters
    parser.add_argument('--epochs', type=int, default=2000)


    # boolean_flag(parser, 'full_rank_gau', default=False)
    parser.add_argument('--decode_type', type=str, default=VAE.GAUSSIAN_FULLRANK)
    parser.add_argument('--optimizer', type=str, default=Optimizer.ADAM)
    # test-obj parameters
    parser.add_argument('--load_date', help='if provided, load the model', type=str, default='')
    parser.add_argument('--load_sub_proj', help='if provided, load the model', type=str, default='hier_ae')
    parser.add_argument('--max_to_keep_save', help='save model every xx iterations', type=int, default=3)
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)

    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add track hyperparameter.
    tester.add_record_param(['info',
                             # 'city', # 'evaluate_percent',
                             'seed',
                             'decode_type',
                             # 'decode_type'
                             'z_size',
                             'learning_rate',
                             "scale_hid",
                             # 'merge_learn',
                             # 'optimizer',
                             'hid_layers',
                             # "split_z_by_layer",
                             "cond_embedding_layer",
                             "constant_z_scale",
                             "cond_with_gau",

                             # 'hid_size'
                             ])
    return args


def vae_data_info_preprocess(env_dict, eval_percent, tester_input):
    features_dict = {}
    feature_name, prepocess_data = None, None
    removed_index = []
    raw_feature_name = None
    days = -1
    for city, env in env_dict.items():
        assert isinstance(env, BaseDriverEnv)
        env._base_reset()
        feature_names = env.features_set
        obs_prepocess_data = env.traj_real_obs.copy()[:, :, :env.dim_static_feature]
        acs_prepocess_data = env.traj_real_acs.copy()
        # acs_feature_name = np.array(feature_names[env.dim_static_feature+1:])
        feature_name = np.array(feature_names)
        raw_feature_name = np.array(feature_names)
        # remove zero fos:
        ac_mean_std = np.concatenate([env.cac_mean_std, env.dac_mean_std], axis=0)
        # ac_mean_std = env.cac_mean_std # np.concatenate([env.cac_mean_std, ], axis=0)
        acs_prepocess_data = acs_prepocess_data * ac_mean_std[:, 1] + ac_mean_std[:, 0]
        acs_prepocess_data = np.round(acs_prepocess_data, 2)
        static_feature_index = []
        for i in range(env.dim_static_feature): # remove whethevr index
            if i not in env.history_fos_index and i not in env.weather_set_index and i not in env.random_scale_index \
                    and i not in env.statistics_index and i not in env.budget_constraint_index and i not in env.UE_index:
                static_feature_index.append(i)
            else:
                removed_index.append(i)
        obs_feature_name = feature_name[static_feature_index]
        feature_name = np.concatenate(
            [obs_feature_name, feature_names[-1 * (env.dim_coupon_feature + env.dim_action):]])
        obs_prepocess_data = obs_prepocess_data[:, :, static_feature_index]
        prepocess_data = np.concatenate([obs_prepocess_data, acs_prepocess_data], axis=2)
        features_dict[city] = prepocess_data
        # days:
        days = prepocess_data.shape[0]

    # zero_std feature:
    zero_std = np.ones(prepocess_data.shape[2]).astype(np.bool)
    # TODO: zero_std feature can be concated with z_var
    for key, value in features_dict.items():
        c_z_std = np.all(value.std(axis=1) < 1e-5, axis=0)
        logger.info("zero std features in city {}".format(key))
        logger.info(np.array(feature_name)[np.where(c_z_std)[0]])
        zero_std = zero_std & c_z_std

    # remove zero features and norm
    logger.info("zero std features")
    logger.info(np.array(feature_name)[np.where(zero_std)[0]])
    feature_name = np.array(feature_name)[np.where(~zero_std)[0]]

    to_remove_index = get_index_from_names(feature_name, to_remove_features)
    # index filter
    # cat filter
    if tester_input.hyper_param['vae_data_type'] == VaeDataType.CAT:
        categorical_index = get_index_from_names(feature_name, categorical_features)
        left_index = categorical_index
    elif tester_input.hyper_param['vae_data_type'] == VaeDataType.NORM:
        # normal filter
        to_remove_index = get_index_from_names(feature_name, to_remove_features)
        to_remove_index += get_index_from_names(feature_name, categorical_features)
        left_index = [x for x in range(feature_name.shape[0]) if x not in to_remove_index]
    elif tester_input.hyper_param['vae_data_type'] == VaeDataType.ALL:
        to_remove_index = get_index_from_names(feature_name, to_remove_features)
        left_index = [x for x in range(feature_name.shape[0]) if x not in to_remove_index]
    else:
        raise NotImplementedError

    if tester_input.hyper_param['vae_task_type'] ==VaeTaskType.STATIC:
        coupon_index = get_index_from_names(feature_name, coupon_features)
        driver_index = get_index_from_names(feature_name, driver_features)
        left_index = [x for x in range(feature_name.shape[0]) if x in left_index and x not in driver_index and x not in coupon_index]
    elif tester_input.hyper_param['vae_task_type'] == VaeTaskType.STATIC_COUPON:
        driver_index = get_index_from_names(feature_name, driver_features)
        left_index = [x for x in range(feature_name.shape[0]) if x in left_index and x not in driver_index]
    elif tester_input.hyper_param['vae_task_type'] == VaeTaskType.ALL:
        pass
    else:
        raise NotImplementedError

    feature_name = feature_name[left_index]

    # low_bound = np.ones(static_feature_name.shape[0]) * np.inf
    # upper_bound = -1 * np.ones(static_feature_name.shape[0]) * np.inf
    normal_index = get_index_from_names(feature_name, acs_normal_distribution_features)
    normal_index += get_index_from_names(feature_name, normal_distribution_features)
    normal_index = sorted(normal_index)
    categorical_index = get_index_from_names(feature_name, categorical_features)
    negative_index = get_index_from_names(feature_name, negative_features)
    remove_min_val_index = get_index_from_names(feature_name, remove_min_val_features)
    coupon_index = get_index_from_names(feature_name, coupon_features)
    driver_index = get_index_from_names(feature_name, driver_features)
    static_feature_index = [x for x in range(feature_name.shape[0]) if x not in coupon_index and x not in driver_index]
    static_feature_index.sort()

    # append condition features
    def index_append(index_list, check_index, new_index):
        if check_index in index_list:
            index_list.append(new_index)

    raw_feature_len = feature_name.shape[0]
    for i in remove_min_val_index:
        feature_name = np.append(feature_name, feature_name[i] + '-bound')
        index_append(coupon_index, i, feature_name.shape[0] - 1)
        index_append(static_feature_index, i, feature_name.shape[0] - 1)
        index_append(driver_index, i, feature_name.shape[0] - 1)

    max_category = 0
    is_min_vale_index = []
    merge_features = np.concatenate([v for k, v in features_dict.items() if k in train_city_list], axis=1)
    merge_features = merge_features[:, :, np.where(~zero_std)[0]]
    merge_features = merge_features[:, :, left_index]
    merge_features[:, :, negative_index] = - merge_features[:, :, negative_index]
    # min as mean for log norm (default distribution)
    if tester_input.hyper_param['decode_type'] == VAE.LOG_GAUSSIAN:
        mean = merge_features.min(axis=(0, 1))
    elif tester_input.hyper_param['decode_type'] == VAE.GAUSSIAN_FULLRANK:
        mean = merge_features.mean(axis=(0, 1))
    else:
        raise NotImplementedError
    mean[normal_index] = merge_features.mean(axis=(0, 1))[normal_index]
    std = merge_features.std(axis=(0, 1))
    min_val = merge_features.min(axis=(0, 1))
    max_val = merge_features.max(axis=(0, 1))
    assert np.any(std) > 0
    if categorical_index == []:
        max_category = 0

    else:
        cat_feature = merge_features[:, :, categorical_index]
        max_category = max(cat_feature.max(axis=(0, 1)) - cat_feature.min(axis=(0,1)))
        mean[categorical_index] = cat_feature.min(axis=(0,1))
        std[categorical_index] = 1

    std[negative_index] = std[negative_index] * -1
    mean[negative_index] = mean[negative_index] * -1
    min_val[negative_index] = merge_features.max(axis=(0, 1))[negative_index] * -1
    max_val[negative_index] = merge_features.min(axis=(0, 1))[negative_index] * -1
    merge_features[:, :, remove_min_val_index] = np.round(merge_features[:, :, remove_min_val_index], 2)
    min_features_val = merge_features[:, :, remove_min_val_index].min(axis=(0, 1))
    condition_index_bound_val = np.zeros(merge_features.shape[-1])
    condition_index_bound_val[remove_min_val_index] = min_features_val
    # min_features = min_features
    # norm_static_features = np.concatenate([merge_features, is_min_features.astype('float32')], axis=-1)
    mean = np.concatenate([mean, np.zeros(list(mean.shape[:-1]) + [len(remove_min_val_index)])], axis=-1)
    std = np.concatenate([std, np.ones(list(mean.shape[:-1]) + [len(remove_min_val_index)])], axis=-1)
    min_val = np.concatenate([min_val, np.zeros(list(mean.shape[:-1]) + [len(remove_min_val_index)])], axis=-1)
    max_val = np.concatenate([max_val, np.ones(list(mean.shape[:-1]) + [len(remove_min_val_index)])], axis=-1)
    x_shape = mean.shape[-1] # norm_static_features.shape[-1]

    for key, value in features_dict.items():
        value = value[:, :, np.where(~zero_std)[0]]
        value = value[:, :, left_index]
        value[:, :, negative_index] = - value[:, :, negative_index]
        value[:, :, remove_min_val_index] = np.round(value[:, :, remove_min_val_index], 2)
        norm_features = value
        min_features = norm_features[:, :, remove_min_val_index]
        is_min_features = min_features_val == min_features
        extend_features = np.concatenate([norm_features, is_min_features.astype('float32')], axis=-1)

        norm_extend_features = (extend_features - mean) / std
        norm_extend_features[:, :, negative_index] = (extend_features[:, :, negative_index] - mean[negative_index] * -1) / std[negative_index] * -1
        is_min_vale_index = list(range(value.shape[-1], x_shape))
        features_dict[key] = [norm_extend_features, mean, std, min_val, max_val]

    merge_features = np.concatenate([v[0] for k, v in features_dict.items() if k in train_city_list], axis=1)
    real_mean = merge_features.min(axis=(0, 1))
    real_std = merge_features.std(axis=(0, 1))
    # upper_bound = upper_bound.astype(np.float32)
    # low_bound = low_bound.astype(np.float32)
    # vae process.
    max_category += 1
    if tester_input.hyper_param['decode_type'] == VAE.LOG_GAUSSIAN:
        log_norm_index = np.ones(x_shape)
        # discrete_index = np.zeros(x_shape)
        # normal_distribution_index = np.zeros(x_shape)
        # condition_map_index = np.zeros(x_shape)
        log_norm_index[categorical_index] = 0
        log_norm_index[normal_index] = 0
        log_norm_index[is_min_vale_index] = 0
        log_norm_index = np.where(log_norm_index == 1)[0]
    elif tester_input.hyper_param['decode_type'] == VAE.GAUSSIAN_FULLRANK:
        normal_index = np.ones(x_shape)
        normal_index[categorical_index] = 0
        if not tester_input.hyper_param['cond_with_gau']:
            normal_index[is_min_vale_index] = 0
        normal_index = np.where(normal_index == 1)[0]
        log_norm_index = []
    else:
        raise NotImplementedError
    # discrete_index[categorical_index] = 1
    # normal_distribution_index[normal_index] = 1
    # condition_map_index[is_min_vale_index] = 1
    condition_map_features = feature_name[remove_min_val_index]


    return x_shape, max_category, raw_feature_len, days, feature_name, condition_map_features, condition_index_bound_val, remove_min_val_index, \
           categorical_index, log_norm_index, normal_index, is_min_vale_index, \
           static_feature_index, coupon_index, driver_index, \
            features_dict, real_mean, real_std

def vae_handle_initial(tester_input, sess, downsample, name,
                       x_shape, max_category, raw_feature_len, days,
                       feature_name, condition_map_features, condition_index_bound_val, remove_min_index,
                       categorical_index, log_norm_index, normal_index, is_min_vale_index,
                       static_feature_index, coupon_index, driver_index,
                       shift, scale, init_ops):

    assert isinstance(tester_input, Tester)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if tester_input.hyper_param['vae_task_type'] == VaeTaskType.STATIC:
            vae_layer_number = 1
        elif tester_input.hyper_param['vae_task_type'] == VaeTaskType.STATIC_COUPON:
            vae_layer_number = 2
        elif tester_input.hyper_param['vae_task_type'] == VaeTaskType.ALL:
            vae_layer_number = 3
        else:
            raise NotImplementedError

        def vae_creator(layer_filter_index, prior_data_index, name, z_size, share_codes, vae_layer_index):
            vae = VAE(hid_size=tester_input.hyper_param['hid_size'], hid_layers=tester_input.hyper_param['hid_layers'],
                      lr=tester_input.hyper_param['learning_rate'], x_shape=[x_shape],
                      z_size=z_size, decode_type=tester_input.hyper_param['decode_type'],
                      log_norm_index=log_norm_index, discrete_index=categorical_index, max_category=max_category,
                      normal_distribution_index=normal_index,
                      condition_distribution_map_index=is_min_vale_index,
                      layer_filter_index=layer_filter_index, share_codes=share_codes,
                      layer_norm=tester_input.hyper_param['layer_norm'], name=name, prior_data_index=prior_data_index,
                      optimizer=tester_input.hyper_param['optimizer'], condition_index_bound_val=condition_index_bound_val,
                      remove_min_index=remove_min_index, split_condition_predict=tester_input.hyper_param['split_condition_predict'],
                      split_z=tester_input.hyper_param['split_z'], full_rank_gau=tester_input.hyper_param['decode_type']==VAE.GAUSSIAN_FULLRANK,
                      merge_dim_likelihood=tester_input.hyper_param['merge_dim_likelihood'],
                      weight_loss=tester_input.hyper_param['weight_loss'], zero_init=tester_input.hyper_param['zero_init'],
                      scale_hid=tester_input.hyper_param['scale_hid'], split_z_by_layer=tester_input.hyper_param['split_z_by_layer'],
                      vae_layer_index=vae_layer_index, cond_embedding_layer=tester_input.hyper_param['cond_embedding_layer'],
                      vae_layer_number=vae_layer_number, constant_z_scale=tester_input.hyper_param['constant_z_scale'],
                      cond_with_gau=tester_input.hyper_param['cond_with_gau'])
            vae.construction()
            return vae
        static_vae = vae_creator(static_feature_index, [], 'static', tester_input.hyper_param['z_size'], None, vae_layer_index=0)
        vae_list =[static_vae]
        if tester_input.hyper_param['vae_task_type'] == VaeTaskType.STATIC_COUPON or tester_input.hyper_param['vae_task_type'] == VaeTaskType.ALL:
            share_codes = static_vae.share_codes
            coupon_acs_vae = vae_creator(coupon_index, static_feature_index, 'coupon', tester_input.hyper_param['z_size'],
                                         share_codes=share_codes, vae_layer_index=1)
            vae_list.append(coupon_acs_vae)
        if tester_input.hyper_param['vae_task_type'] == VaeTaskType.ALL:
            driver_acs_vae = vae_creator(driver_index, static_feature_index + coupon_index, 'driver', tester_input.hyper_param['z_size'],
                                         share_codes=share_codes, vae_layer_index=2)
            vae_list.append(driver_acs_vae)
        # [static_vae, coupon_acs_vae, driver_acs_vae]
    vae_hander = VaeHander(vae_list=vae_list, name=name,
                           bandwidth_coef=tester_input.hyper_param['bandwidth'], x_shape=x_shape,
                           raw_feature_len=raw_feature_len, downsample=downsample, days=days,
                           feature_name=feature_name, condition_map_features=condition_map_features,
                           sess=sess, city_list=city_list, merge_learn=tester_input.hyper_param['merge_learn'],
                           shift=shift, scale=scale, init_ops=init_ops)
    return vae_hander


def main(args):
    from common.tester import tester
    sess = U.make_session(num_cpu=16).__enter__()
    set_global_seeds(args.seed)
    task_name = tester.task_gen([args.vae_task_type, args.vae_data_type, args.task])
    tester.configure(task_name, 'run_driver', add_record_to_pkl=False,
                     root=args.log_root,
                     max_to_keep=args.max_to_keep_save)
    if args.load_date is not '':
        import common
        tester = tester.load_tester(record_date=args.load_date, prefix_dir=args.load_sub_proj,
                                                  log_root=args.log_root)
        tester.init_unserialize_obj(sess, )
        common.tester.tester = tester

    tester.print_args()
    if args.vae_task_type == VaeTaskType.STATIC:
        downsample = 15
    else:
        downsample = 5
    # load env set.
    city_env_dict = {}
    for city in city_list:
        logger.info("extract city {}".format(city))

        def env_create_fn(env_name, folder_name, expert_path, city, start_date, end_date, ):
            return BaseDriverEnv(env_name, folder_name, expert_path, city, start_date, end_date,
                                 delay_date=False, evaluate_percent=1, driver_type_amount=2,
                                 trajectory_batch=9192, reduce_param=True,
                                 update_his_fo=True, constraint_driver_action=False,
                                 partition_data=1, norm_obs=False)

        env = env_create_fn('', city + args.folder_name, args.expert_path, city, args.start_date, args.end_date)
        city_env_dict[city] = env
    # preprocess dataset.
    x_shape, max_category, raw_feature_len, days, feature_name, condition_map_features, condition_index_bound_val, remove_min_index, \
    categorical_index, log_norm_index, normal_index, is_min_vale_index, \
    static_feature_index, coupon_index, driver_index, features_dict, real_mean, real_std = vae_data_info_preprocess(city_env_dict,
                                                                                                                    eval_percent=1, tester_input=tester)
    # create vae_handler
    shift = features_dict[train_city_list[0]][1]
    scale = features_dict[train_city_list[0]][2]
    vae_handler = vae_handle_initial(tester, sess, downsample, 'distribution_emb',
                       x_shape, max_category, raw_feature_len, days,
                       feature_name, condition_map_features, condition_index_bound_val, remove_min_index,
                       categorical_index, log_norm_index, normal_index, is_min_vale_index,
                       static_feature_index, coupon_index, driver_index, shift, scale, init_ops=True)
    logger.info("start initialize")
    U.initialize()
    tester.new_saver('distribution_emb')
    if args.load_date is not '':
        start_epoch = vae_handler.load_from_checkpoint(tester.checkpoint_dir, target_prefix_name='distribution_emb/')
        # start_epoch = tester.load_checkpoint()
    else:
        start_epoch = 0
    # tester.add_graph(sess)
    # tester.save_tensor_graph(sess)
    epochs = args.epochs
    log_epoch = int(epochs / 20)
    log_epoch = max(5, log_epoch)
    code_min = code_max = 0
    min_predict_error = 10000
    for epoch in range(start_epoch, epochs):
        logger.info("start learning epoch {}".format(epoch))
        tester.time_step_holder.set_time(epoch)
        if epoch % 20 == 0 and epoch >= 0:
            plot_pic = epoch % log_epoch == 0
            if plot_pic:
                city_index = 0
                fig, ax = plt.subplots(nrows=int(days / downsample) + 1, ncols=len(city_list) + 1,
                                       figsize=(6 * len(city_list), 6 * (days / downsample)))

                x = np.arange(vae_handler.z_size)
            for test_city in test_city_list:
                ret_codes = vae_handler.evaluation(real_data_info=features_dict[test_city],
                                             test_city=test_city,
                                             prefix_file_name='test',
                                             epoch=epoch, plot_pic=plot_pic)
                if np.asarray(ret_codes).shape[0] != 0:
                    code_min = np.min([code_min, np.min(np.array(ret_codes))])
                    code_max = np.max([code_max, np.max(np.array(ret_codes))])
                    print("{}, {}".format(code_min, code_max))
                    if plot_pic:
                        for d, code in enumerate(ret_codes):
                            cur_ax = ax[d, city_index]
                            cur_ax.plot(x, code, "*--")
                            # cur_ax.hist(code, bins=20, density=False, histtype='step',)
                            cur_ax.set_ylim(code_min, code_max)
                            cur_ax.set_title("z_emb: {}(test)-{}".format(test_city, d * downsample))
                    city_index += 1
            for test_city in train_city_list:
                ret_codes = vae_handler.evaluation(real_data_info=features_dict[test_city],
                                             test_city=test_city,
                                             prefix_file_name='train',
                                             epoch=epoch, plot_pic=plot_pic, )

                if np.asarray(ret_codes).shape[0] != 0:
                    code_min = np.min([code_min, np.min(np.array(ret_codes))])
                    code_max = np.max([code_max, np.max(np.array(ret_codes))])
                    if plot_pic:
                        for d, code in enumerate(ret_codes):
                            cur_ax = ax[d, city_index]
                            cur_ax.plot(x, code, "*--")
                            cur_ax.set_ylim(code_min, code_max)
                            cur_ax.set_title("z_emb: {}(train)-{}".format(test_city, d * downsample))
                    city_index += 1
            # embedding predict learning
            predict_error = vae_handler.embedding_predictable_evaluation(features_dict)
            if min_predict_error > predict_error:
                min_predict_error = predict_error
                tester.save_checkpoint(epoch)
            vae_handler.save_kde_map()
            if plot_pic:
                os.makedirs("{}/{}/".format(tester.results_dir, 'z_emb'), exist_ok=True)
                fig.savefig('{}/{}/vae-{}.png'.format(tester.results_dir, 'z_emb', epoch), dpi=60)
                plt.close('all')
                code_min = code_max = 0
        for _ in range(10):
            train_number = np.random.randint(len(train_city_list))
            train_city = train_city_list[train_number]
            train_days = np.random.randint(days)
            def train():
                for _ in range(10):
                    next_batch_data = features_dict[train_city][0]
                    next_batch_data = next_batch_data[train_days]
                    shu_idx = np.arange(next_batch_data.shape[0])
                    np.random.shuffle(shu_idx)
                    next_batch_data = next_batch_data[shu_idx[:args.batch_size]]
                    vae_handler.train(next_batch_data)
            tester.time_used_wrap("train vae -{}".format(epoch), train)


if __name__ == '__main__':
    args = argsparser()
    main(args)
