import gym
import sys
sys.path.append('../../')
import argparse
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds, tf_util as U
from transfer_learning.src.private_config import *
from common.tester import tester
from common.env_base import BaseDriverEnv
from transfer_learning.src.driver_simenv_disc import DriverEnv, MultiCityDriverEnv
from common.mlp_policy import MlpPolicy as DemerMlp
import numpy as np
from transfer_learning.src.policies import EnvExtractorPolicy, EnvAwarePolicy
from ppo2.policies import MlpPolicy as PpoMlp
from ppo2.ppo2 import PPO2 as ppo2_vanilla
from transfer_learning.src.env_aware_ppo2 import PPO2 as ppo2_env_aware
from trpo_mpi.trpo_mpi import TRPO as trpo_vanilla
import tensorflow as tf
from transfer_learning.src.func import *
from common.config import *
from common import logger


def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    # tester configuration (for log)
    parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--task', help='environment ID', default='')
    parser.add_argument('--trans_type', help='environment ID', default=TransType.DIRECT)
    parser.add_argument('--hand_code_type', help='environment ID', default=HandCodeType.STD_TRANS)
    parser.add_argument('--alo_type', help='environment ID', default=AlgorithmType.PPO)
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    parser.add_argument('--load_date', type=str, default='')
    parser.add_argument('--load_sub_proj', type=str, default='transfer')
    parser.add_argument('--max_to_keep_save', help='save model every xx iterations', type=int, default=3)
    boolean_flag(parser, 'full_tensorboard_log', default=False)
    boolean_flag(parser, 'save_checkpoint', default=False)
    # data load configuration
    parser.add_argument('--start_date', type=int, default=20190601)
    parser.add_argument('--city', type=str, default='Heyuan')
    parser.add_argument('--end_date', type=int, default=20190630)
    parser.add_argument('--expert_path', type=str, default=DATA_SET_ROOT)
    parser.add_argument('--folder_name', type=str, default='_ceil_reduce_single_day_coupon_29')
    # dfm model
    boolean_flag(parser, 'data_only', default=False)
    # env configuration
    boolean_flag(parser, 'constraint_driver_action', default=True)
    boolean_flag(parser, 'kl_penalty', default=False)
    parser.add_argument('--coupon_ratio_scale', type=float, default=1.0)
    parser.add_argument('--reward_scale', type=float, default=15.0)
    parser.add_argument('--per_gmv', type=float, default=20.0)
    parser.add_argument('--partition_data', type=float, default=0.01)
    boolean_flag(parser, 'delay_date', default=True)
    boolean_flag(parser, 'deter_env', default=False)
    boolean_flag(parser, 'threshold_reward', default=True)
    boolean_flag(parser, 'zero_cp_penalty', default=False)

    parser.add_argument('--cluster_type',  default=ClusterType.MAP)
    parser.add_argument('--cluster_trace_days', type=float, default=3)
    boolean_flag(parser, 'unit_cost', default=True)
    boolean_flag(parser, 'only_gmv', default=False)
    boolean_flag(parser, 'adv_rew', default=False)
    boolean_flag(parser, 'given_scale', default=False)
    boolean_flag(parser, 'given_statistics', default=True)
    boolean_flag(parser, 'hand_code_driver_action', default=False)
    boolean_flag(parser, 'mdp_env', default=True)
    boolean_flag(parser, 'remove_hist_state', default=False)
    parser.add_argument('--time_change_coef', type=float, default=1e-6)
    parser.add_argument('--scale_coef', type=float, default=0.2)  # useless
    parser.add_argument('--driver_weight', type=float, default=1.0)
    parser.add_argument('--group_weight', type=float, default=10)
    parser.add_argument('--weight_scale', type=float, default=10.0)
    # demer configuration
    parser.add_argument('--policy_hidden_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--head_amount', type=int, default=2)
    boolean_flag(parser, 'confounder', default=True)
    boolean_flag(parser, 'cp_unit_mask', default=True)
    boolean_flag(parser, 'simp_cluster', default=True)
    # learning configuration
    parser.add_argument('--noptepochs', type=int, default=3)  # modified
    boolean_flag(parser, 'norm_obs', default=False)  # hurt perf, check bug.
    boolean_flag(parser, 'simp_state', default=True)
    boolean_flag(parser, 'remove_done', default=True)
    boolean_flag(parser, 'budget_constraint', default=True)
    parser.add_argument('--budget_expand', type=float, default=1.5)
    boolean_flag(parser, 'UE_penalty', default=True)
    boolean_flag(parser, 'gmv_rescale', default=True)
    boolean_flag(parser, 'merge_city_samples', default=True)
    boolean_flag(parser, 'rms_opt', default=False)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lstm_train_freq', type=int, default=1)
    parser.add_argument('--soft_update_freq', type=int, default=1)
    # v: 200, p: 20 seems to be better
    parser.add_argument('--v_grad_norm', type=float, default=0.5)
    parser.add_argument('--p_grad_norm', type=float, default=0.5)
    parser.add_argument('--lda_max_grad_norm', type=float, default=0.1)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=10e8)
    parser.add_argument('--batch_timesteps', help='number of timesteps per batch', type=int, default=10000) # reduce network and increase batch
    # parser.add_argument('--batch_size', help='number of timesteps per batch', type=int, default=4096)
    parser.add_argument('--keep_dids_times', help='number of timesteps per batch', type=int, default=1)
    # policy network parameters
    parser.add_argument('--policy_layers', type=int, default=[128, 128, 128, 32], nargs='+') #[512, 128]
    parser.add_argument('--policy_act', type=str, default=ActFun.LEAKY_RELU)
    parser.add_argument('--ent_coef', type=float, default=1e-5)
    parser.add_argument('--v_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_env_learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_rescale', type=float, default=1.0)
    boolean_flag(parser, 'redun_distri', default=True)

    parser.add_argument('--policy_init_scale', type=float, default=0.01)
    # TODO: Why this doesn't work on all value selected?
    parser.add_argument('--lr_remain_ratio', type=float, default=0.01) # 0.1 会使得性能相对稳定， 0.005和0.001则： [人造]有所下降；[demer] 更为稳定
    parser.add_argument('--l2_reg_pi', type=float, default=1e-6)
    parser.add_argument('--l2_reg_v', type=float, default=1e-6)
    boolean_flag(parser, 'learn_var', default=False) # TODO: learning variance seems to be better but unstable.
    boolean_flag(parser, 'double_city', default=False)
    boolean_flag(parser, 'use_target_city', default=False)
    boolean_flag(parser, 'normalize_returns', default=False)
    boolean_flag(parser, 'normalize_rew', default=True)
    boolean_flag(parser, 'hand_code_distribution', default=False)
    boolean_flag(parser, 'just_scale', default=True)
    boolean_flag(parser, 'res_net', default=False)
    boolean_flag(parser, 'layer_norm', default=True)
    # env extractor network parameters
    parser.add_argument('--l2_reg_env', type=float, default=1e-6)
    parser.add_argument('--driver_cluster_days', type=int, default=3) # driver_cluster_days = 7 hurt performance.
    parser.add_argument('--env_params_size', type=int, default=48)
    parser.add_argument('--consistent_ecoff', type=float, default=0.8)
    parser.add_argument('--samples_per_driver', type=int, default=5)
    parser.add_argument('--tau', type=int, default=0.001)
    parser.add_argument('--init_scale_output', type=float, default=0.000001)
    parser.add_argument('--env_extractor_layers', type=int, default=[128, 128], nargs='+')
    parser.add_argument('--n_lstm', type=int, default=512)
    parser.add_argument('--clus_param', type=int, default=[4, 3, 3, 2], nargs='+')
    boolean_flag(parser, 'use_lda_loss', default=True)
    boolean_flag(parser, 'use_resnet', default=False)
    boolean_flag(parser, 'stop_critic_gradient', default=True)
    boolean_flag(parser, 'log_lda', default=False)
    boolean_flag(parser, 'scaled_lda', default=True)
    boolean_flag(parser, 'cliped_lda', default=False)
    boolean_flag(parser, 'square_lda', default=True)
    parser.add_argument('--constraint_lda', type=str, default=ConstraintLDA.Proj2CGC)
    boolean_flag(parser, 'use_cur_obs', default=True)
    boolean_flag(parser, 'env_relu_act', default=True)
    boolean_flag(parser, 'env_out_relu_act', default=False)
    boolean_flag(parser, 'lstm_layer_norm', default=False)
    boolean_flag(parser, 'env_layer_norm', default=True)
    boolean_flag(parser, 'lstm_critic', default=True)
    # distribution embedding parameters
    boolean_flag(parser, 'stable_dist_embd', default=True)
    boolean_flag(parser, 'use_distribution_info', default=True)
    boolean_flag(parser, 'vae_test', default=False)
    # concerned parameters
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--evaluate_percent', type=float, default=1)
    parser.add_argument('--coupon_rate', type=float, default=0.07)  # auto compute by historical data?
    boolean_flag(parser, 'eval_stochastic_policy', default=False)
    boolean_flag(parser, 'disc_env', default=True)
    parser.add_argument('--sim_version', type=int, default=-1)

    ## discretization
    boolean_flag(parser, 'high_rew', default=True)
    boolean_flag(parser, 'revise_step', default=False) # 是否运行基于专家知识的修正的动作
    boolean_flag(parser, 'use_predict_dist', default=True)
    boolean_flag(parser, 'use_value_weight', default=False)
    boolean_flag(parser, 'use_restore_act', default=False)
    boolean_flag(parser, 'use_predict_q', default=False)
    parser.add_argument('--disc_tdgamma', type=float, default=0.9)
    parser.add_argument('--disc_lam', type=float, default=0.97)
    parser.add_argument('--cluster_number', type=int, default=20)
    parser.add_argument('--disc_number', type=float, default=200)
    parser.add_argument('--disc_percent', type=float, default=0.99)
    parser.add_argument('--disc_precision', type=float, default=5)
    parser.add_argument('--disc_epi_num', type=int, default=20)

    args = parser.parse_args()
    # if args.final_learning_rate > args.learning_rate:
    #     args.final_learning_rate = args.learning_rate
    args.v_learning_rate *= args.lr_rescale
    args.p_learning_rate *= args.lr_rescale
    args.p_env_learning_rate *= args.lr_rescale
    if args.trans_type == TransType.DIRECT or args.trans_type == TransType.DIRECT_TRANS or args.trans_type == TransType.DIRECT_TRANS_CITY\
            or args.trans_type == TransType.UNIVERSAL:
        # args.final_learning_rate = 3e-5
        # args.p_learning_rate = args.g_learning_rate = 3e-4
        # args.p_grad_norm = args.p_grad_norm = 20
        # args.noptepochs = 10
        args.use_distribution_info = False
    if args.trans_type == TransType.ENV_AWARE_VAN:
        args.use_distribution_info = False
        args.use_lda_loss = False
    if args.trans_type == TransType.ENV_AWARE_DIRECT:
        args.use_distribution_info = False

    if not args.hand_code_driver_action:
        args.simp_state = False
        # temp remove
        # args.budget_constraint = True
        # args.UE_penalty = False

        args.unit_cost = True
        args.given_statistics = False
        args.given_scale = False

        args.remove_hist_state = False
        args.only_gmv = False
        args.cluster_type = ClusterType.FOS # demer 环境中，认为司机是按照完单进行聚类的
    else:
        args.budget_constraint = False
        args.unit_cost = False
        args.UE_penalty = False
        args.simp_state = True
        args.given_statistics = True
        args.constraint_driver_action = False
        args.cp_unit_mask = False
        args.only_gmv = True
        args.remove_hist_state = True
        args.cluster_type = ClusterType.MAP  # demer 环境中，司机类型是随机生成的，但是影响完单量
    args.cp_unit_mask = False
    args.constraint_driver_action = False

    if args.hand_code_driver_action and args.simp_cluster:
        ClusterFeatures.ORDER = [ConstraintType.MAX_STANDARD_FOS, ConstraintType.MEAN_STANDARD_FOS]
        args.clus_param = args.clus_param[:2]
    # if args.lr_remain_ratio:
    #     args.lr_remain_ratio = args.lr_remain_ratio / (args.num_timesteps / 10e8)
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add track hyperparameter.
    return args

# Custom MLP policy of three layers of size 128 each

if __name__ == '__main__':

    args = argsparser()
    if args.city == 'all':
        test_cities = city_list
    else:
        test_cities = [args.city]
    for tc in test_cities:
        kwargs = vars(args)
        set_global_seeds(args.seed)
        tester.set_hyper_param(**kwargs)
        tester.hyper_param['city'] = tc
        # add track hyperparameter.
        tester.clear_record_param()
        tester.add_record_param(['info',
                                 'trans_type',
                                 # "given_scale",
                                 "ent_coef",
                                 # "weight_scale",
                                 # 'city',
                                 # "lam",
                                 'seed',
                                 # "policy_layers",
                                 # "env_params_size",
                                 # "p_env_learning_rate",
                                 "batch_timesteps",
                                 # "tau",
                                 "policy_layers",
                                 # "budget_constraint",
                                 "lstm_critic",
                                 "lr_rescale",
                                 # "UE_penalty",
                                 # "data_only",
                                 # "unit_cost",
                                 # "samples_per_driver",
                                 # "samples_per_driver",
                                 # "init_scale_output",
                                 # "lr_rescale",
                                 # "remove_hist_state",
                                 # "use_lda_loss",
                                 # "lstm_train_freq",
                                 # "square_lda",
                                 "scaled_lda",
                                 "constraint_lda",
                                 # "scaled_lda",
                                 # "env_extractor_layers",
                                 # "n_lstm",
                                 # "cliped_lda",
                                 # "redun_distri",
                                 # "env_params_size",
                                 # "env_extractor_layers",
                                 # "lr_remain_ratio",
                                 # "lstm_layer_norm",
                                 # "soft_update_freq",
                                 # "budget_constraint",
                                 # "UE_penalty"

                                 ])

        sess = U.make_session(num_cpu=16).__enter__()
        # import dataset
        def task_name_gen():
            task_name = '-'.join([tc, "ar-" + str(args.adv_rew), args.task])
            if args.hand_code_driver_action:
                task_name += 'hca13-g-{}-d-{}'.format(args.group_weight, args.driver_weight)
                if args.time_change_coef != 0.1:
                    task_name += 'tg-{}'.format(args.time_change_coef)
                # if args.given_scale:
                #     task_name += 'gs-'
                if args.given_statistics:
                    task_name += 'gstat-'
                if args.mdp_env:
                    task_name += 'mdp-'
            else:
                task_name += 'demer1-'
                if args.kl_penalty:
                    task_name += 'kl-'
            if not args.only_gmv:
                if args.unit_cost:
                    task_name += 'unit_cost-'
                else:
                    if args.coupon_ratio_scale > 1.0:
                        task_name += 'cprs_2_{}'.format(args.coupon_ratio_scale)
            else:
                task_name += 'gmv-'
            if args.constraint_driver_action:
                task_name += 'cda-'
            return task_name
        tester.configure(task_name_gen(), 'run_trans', record_date=args.load_date,
                         add_record_to_pkl=False, root=args.log_root,
                         max_to_keep=args.max_to_keep_save)
        # if args.load_date is not '':
        #     tester = tester.load_tester(args.load_date, prefix_dir='direct', log_root=args.log_root)
        tester.print_args()

        def env_create_fn(env_name, folder_name, expert_path, city, start_date, end_date, load_model_path, batch_timesteps):
            # parser.add_argument('--time_change_coef', type=float, default=0.1)
            # parser.add_argument('--scale_coef', type=float, default=0.1)
            # parser.add_argument('--driver_coef', type=float, default=0.5)
            # parser.add_argument('--group_coef', type=float, default=1.0)
            return DriverEnv(name=env_name, folder_name=folder_name, expert_path=expert_path, city=city,
                              start_date=start_date, end_date=end_date, policy_fn=policy_fn,
                             hand_code_type=args.hand_code_type,
                              # original PG params.
                              coupon_ratio_scale=args.coupon_ratio_scale,
                              partition_data=args.partition_data,
                              re_construct_ac=True, sim_version=-1,  load_model_path=load_model_path,
                              update_his_fo=True, reduce_param=True, # Demer params.
                              eval_result_dir=tester.results_dir, delay_date=args.delay_date,
                              driver_type_amount=args.head_amount,
                              trajectory_batch=batch_timesteps,
                              constraint_driver_action=args.constraint_driver_action,
                              rew_scale=args.reward_scale, adv_rew=args.adv_rew, unit_cost_penlaty=args.unit_cost,
                             hand_code_driver_action=args.hand_code_driver_action, only_gmv=args.only_gmv,
                             clus_param=args.clus_param, time_change_coef=args.time_change_coef,
                             scale_coef=args.scale_coef, driver_coef=None,
                             group_coef=None, given_scale=args.given_scale,
                             given_statistics=args.given_statistics, cluster_type=args.cluster_type,
                             norm_obs=args.norm_obs, mdp_env=args.mdp_env,
                             simp_state=args.simp_state, deter_env=args.deter_env, kl_penalty=args.kl_penalty,
                             UE_penalty=args.UE_penalty, budget_constraint=args.budget_constraint,
                             cp_unit_mask=args.cp_unit_mask, samples_per_driver=args.samples_per_driver,
                             gmv_rescale=args.gmv_rescale, remove_hist_state=args.remove_hist_state,
                             budget_expand=args.budget_expand)
                              # remove: reduce param; update_his_fo

        def policy_fn(name, ob_space, ac_space, cac_space, reuse=False):
            return DemerMlp(name=name, ob_space=ob_space, ac_space=ac_space, cac_space=cac_space,
                                        reuse=reuse, hid_size=args.policy_hidden_size,
                                        num_hid_layers=args.hidden_layers,
                                        cms_min=env.cac_min_value, dms_min=env.dac_min_value,
                                        cms_max=env.cac_max_value, dms_max=env.dac_max_value, confounder=args.confounder,
                                        head_amount=args.head_amount,
                                        gaussian_fixed_var=False, coupon_hid_layers=2, tanh_oup=True)

        # auto generate city set.
        if tc == '':
            test_city_size = 1
            relation_matrix = tester.load_pickle(relation_matrix_file_name)
            test_city_list, train_city_list = auto_train_test_split(city_list, relation_matrix, test_city_size)

        else:
            test_city_list = [tc + '-test']
        if args.trans_type == TransType.DIRECT or args.trans_type == TransType.ENV_AWARE_DIRECT:
            train_city_list = test_city_list
        elif args.trans_type == TransType.DIRECT_TRANS:
            train_city_list = [tc[:-5] for tc in test_city_list]
        elif args.trans_type == TransType.DIRECT_TRANS_CITY:
            relation_matrix = tester.load_pickle(relation_matrix_file_name)
            tc_index = city_list.index(tc)
            tc_relation = relation_matrix[tc_index, ...].mean(axis=(1, 2))
            train_city_selected = city_list[tc_relation.argsort()[-1]] # select the biggest difference
            train_city_list = [train_city_selected]
            # test_city_list, train_city_list = auto_train_test_split(city_list, relation_matrix, 1)

        elif args.trans_type == TransType.UNIVERSAL or args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN:
            if args.double_city:
                train_city_list = [c for c in city_list if c + '-test' not in test_city_list]
                train_city_list += [c + '-test' for c in city_list if c + '-test' not in test_city_list]
            else:
                train_city_list = [c for c in city_list if c + '-test' not in test_city_list]
            if args.use_target_city:
                train_city_list += [c for c in city_list if c + '-test' in test_city_list]
        else:
            raise NotImplementedError

        logger.info("gen train city {}".format(train_city_list))
        logger.info("gen test city {}".format(test_city_list))
        if args.merge_city_samples:
            batch_timesteps = int(args.batch_timesteps / len(train_city_list))
        else:
            batch_timesteps = args.batch_timesteps
        logger.info("batch_timesteps per city {}".format(batch_timesteps))
        city_env_dict = {}
        for data_range, city, folder_name, checkpoint_date in zip(demer_date_range, city_list, demer_folder_name,
                                                                  demer_checkpoint_date):
            dir, load_model_path = tester.load_checkpoint_from_date(checkpoint_date, prefix_dir='train',
                                                                    log_root=args.log_root + '../model/')
            assert city in load_model_path, "city {}, path {}".format(city, load_model_path)
            env = env_create_fn(city, folder_name, args.expert_path, city, data_range[0], data_range[1],
                                dir + '/' + load_model_path, batch_timesteps)
            if not args.hand_code_driver_action:
                env.load_simulator()
            city_env_dict[city] = env

        for data_range, city, folder_name, checkpoint_date in zip(demer_test_date_range, city_list, demer_folder_name,
                                                                  demer_test_checkpoint_date):
            dir, load_model_path = tester.load_checkpoint_from_date(checkpoint_date, prefix_dir='train',
                                                                    log_root=args.log_root + '../model/')
            assert city in load_model_path, "city {}, path {}".format(city, load_model_path)
            env = env_create_fn(city + '-test', folder_name, args.expert_path, city, data_range[0], data_range[1],
                                dir + '/' + load_model_path, batch_timesteps)
            if not args.hand_code_driver_action:
                env.load_simulator()
            city_env_dict[city + '-test'] = env
        env = MultiCityDriverEnv(city_env_dict=city_env_dict, city_list=train_city_list,
                                representation_city_name=test_city_list[0], cp_unit_mask=args.cp_unit_mask)
        eval_env = MultiCityDriverEnv(city_env_dict=city_env_dict, city_list=test_city_list,
                                representation_city_name=test_city_list[0], cp_unit_mask=args.cp_unit_mask)

        # budget error
        # for city in city_list:
        #     inter_did = np.intersect1d(city_env_dict['{}-test'.format(city)].select_dids,
        #                                city_env_dict[city].select_dids)
        #     test_env_filter_did = np.where(np.isin(city_env_dict['{}-test'.format(city)].select_dids, inter_did))
        #     env_filter_did = np.where(np.isin(city_env_dict[city].select_dids, inter_did))
        #     env_filter_budget = city_env_dict[city].driver_budget[env_filter_did]
        #     test_env_filter_budget = city_env_dict['{}-test'.format(city)].driver_budget[env_filter_did]
        #     budget_error = np.abs(env_filter_budget - test_env_filter_budget)
        #     print("error to {} - {}".format(city, np.sum(budget_error) / np.sum(env_filter_budget)))
        #     print("error to {} - {}".format(city + "-test", np.sum(budget_error) / np.sum(test_env_filter_budget)))
        vae_handler = None
        # create vae
        # load tester
        vae_tester = tester.load_tester(vae_checkpoint_date, prefix_dir='all-norm-hier_ae',
                                        log_root=args.log_root + '../vae/')
        # load vae
        dir, load_model_path = vae_tester.load_checkpoint_from_date(vae_checkpoint_date,
                                                                    prefix_dir='all-norm-hier_ae',
                                                                    log_root=args.log_root + '../vae/')
        vae_checkpoint_path = dir + '/' + load_model_path
        from vae.src.run_hier_ae_fix import vae_handle_initial, vae_data_info_preprocess

        x_shape, max_category, raw_feature_len, days, feature_name, condition_map_features, condition_index_bound_val, remove_min_index, \
        categorical_index, log_norm_index, normal_index, is_min_vale_index, \
        static_feature_index, coupon_index, driver_index, features_dict, real_mean, real_std = vae_data_info_preprocess(city_env_dict, 1, vae_tester)

        if args.hand_code_driver_action:
            feature_statistics = None
            np.random.seed(20)
            for city in city_list:
                features = features_dict[city]
                feature_statistics = np.concatenate([features[0].mean(axis=(0, 1)),
                                                     features[0].std(axis=(0, 1))], axis=0)
                city_env_dict[city].feature_statistics = feature_statistics * 10
                features = features_dict[city + '-test']
                feature_statistics = np.concatenate([features[0].mean(axis=(0, 1)), features[0].std(axis=(0, 1))], axis=0)
                city_env_dict[city + '-test'].feature_statistics = feature_statistics * 10
            # feature statistics filter (std > 0.1)
            feature_statistics_list = []
            for city in city_list:
                if city_env_dict[city].feature_statistics is not None:
                    feature_statistics_list.append(city_env_dict[city].feature_statistics)
                if city_env_dict[city + '-test'].feature_statistics is not None:
                    feature_statistics_list.append(city_env_dict[city + '-test'].feature_statistics)
            feature_statistics_list = np.asarray(feature_statistics_list)
            feature_filter_index = feature_statistics_list.std(axis=0) > 0.1
            for city in city_list:
                if city_env_dict[city].feature_statistics is not None:
                    city_env_dict[city].feature_statistics = city_env_dict[city].feature_statistics[feature_filter_index]
                if city_env_dict[city + '-test'].feature_statistics is not None:
                    city_env_dict[city + '-test'].feature_statistics = city_env_dict[city + '-test'].feature_statistics[
                    feature_filter_index]
            s_shape = city_env_dict[train_city_list[0]].feature_statistics.shape[0]
            print("s shape {}, original_shape {}".format(s_shape, feature_filter_index.shape[0]))
            np.random.seed(20)
            b = np.random.normal(0, 1, size=(s_shape, ConstraintType.NUMBER))
            w = np.random.normal(0, 1, size=(s_shape, ConstraintType.NUMBER))
            # cluster_shape = np.prod(args.clus_param)
            # cluster_matrix_w = np.random.normal(-1, 1, size=(cluster_shape, ConstraintType.NUMBER))
            # cluster_matrix_b = np.random.normal(-1, 1, size=(cluster_shape, ConstraintType.NUMBER))
            # cluster_shape = np.prod(args.clus_param)
            clus_param = args.clus_param
            hand_code_clus_type = np.array([ConstraintType.MAX_STANDARD_FOS, ConstraintType.MEAN_STANDARD_FOS])
            hand_code_clus_type_index = []
            for hcct in hand_code_clus_type:
                hand_code_clus_type_index.append(ConstraintType.ORDER.index(hcct))
            hand_code_clus_type_index = np.array(hand_code_clus_type_index)
            assert hand_code_clus_type_index.shape[0] == 2
            max_number = clus_param[hand_code_clus_type_index[0]] + 2
            mean_number = clus_param[hand_code_clus_type_index[1]] + 2
            linear_coeff_max = np.expand_dims(np.arange(0.1, 2.2, 1 /max_number)[:max_number], 0)
            linear_coeff_mean = np.expand_dims(np.arange(0.1, 2.2, 1 / mean_number)[:mean_number], 0)
            cluster_matrix_w = np.repeat(np.expand_dims(np.prod([linear_coeff_max, linear_coeff_mean.T]), 2),
                                         axis=2, repeats=ConstraintType.NUMBER)

            cluster_matrix_b = cluster_matrix_w
            driver_matrix_meta_w = np.random.uniform(0, 1, size=(len(env.representative_city.to_zero_state_index), ConstraintType.NUMBER))
            driver_matrix_meta_b = np.random.uniform(0, 1, size=(len(env.representative_city.to_zero_state_index), ConstraintType.NUMBER))

            random_scale_weight = {}
            random_scale_bias = {}

            # auto generate coef
            groups_b = []
            rand_drivers_b = []
            for city in city_list:
                eval_city = city_env_dict[city]
                group_b = np.matmul(np.expand_dims(eval_city.feature_statistics, axis=0), b)
                norm_obs = eval_city._norm_obs(eval_city.traj_real_obs)[:, :, eval_city.to_zero_state_index]
                driver_b = np.matmul(norm_obs, driver_matrix_meta_b)
                rand_drivers_b.append(driver_b)
                groups_b.append(group_b)
                eval_city = city_env_dict[city + '-test']
                group_b = np.matmul(np.expand_dims(eval_city.feature_statistics, axis=0), b)
                groups_b.append(group_b)
                norm_obs = eval_city._norm_obs(eval_city.traj_real_obs)[:, :, eval_city.to_zero_state_index]
                driver_b = np.matmul(norm_obs, driver_matrix_meta_b)
                rand_drivers_b.append(driver_b)

            group_weight = args.group_weight
            auto_group_coef = group_weight / np.array(groups_b).max(axis=0)
            if args.cluster_type == ClusterType.FOS or args.cluster_type == ClusterType.MAP:
                auto_driver_coef = args.driver_weight
            elif args.cluster_type == ClusterType.RAND:
                auto_driver_coef = args.driver_weight / np.array([rb.max(axis=(0, 1)) for rb in rand_drivers_b]).max(
                    axis=0)
            else:
                raise NotImplementedError
            scale_coef = args.weight_scale / (group_weight + args.driver_weight)
            for city in city_list:
                random_scale_weight[city + '-test'] = random_scale_weight[city] = driver_matrix_meta_w
                random_scale_bias[city + '-test'] = random_scale_bias[city] = driver_matrix_meta_b

                def random_weight_gen(city):
                    select_city = city_env_dict[city]
                    select_city.hand_code_clus_type_index = hand_code_clus_type_index
                    select_city.hand_code_clus_type = hand_code_clus_type
                    if args.hand_code_type == HandCodeType.MAX_TRANS:
                        select_city.cac_max_value[0:2] *= 100
                    else:
                        select_city.cac_max_value[0:2] *= 5
                    select_city.group_matrix_b = b
                    select_city.group_matrix_w = w
                    select_city.cluster_matrix_w = cluster_matrix_w
                    select_city.cluster_matrix_b = cluster_matrix_b
                    select_city.driver_matrix_w = random_scale_weight[city]
                    select_city.driver_matrix_b = random_scale_bias[city]
                    if args.cluster_type == ClusterType.MAP:
                        global_driver_w = np.random.uniform(0, 1, size=(select_city.driver_number, ConstraintType.NUMBER))
                        global_driver_b = np.random.uniform(0, 1, size=(select_city.driver_number, ConstraintType.NUMBER))
                        select_city.driver_w = global_driver_w
                        select_city.driver_b = global_driver_b
                    select_city.group_coef = auto_group_coef
                    select_city.driver_coef = auto_driver_coef
                    select_city.scale_coef = scale_coef
                    select_city.hand_code_info_gen()
                random_weight_gen(city)
                random_weight_gen(city + '-test')

            for key, rs in random_scale_weight.items():
                print(key, ":", rs[:4, 4])
            np.random.seed(args.seed)
            # eval_env.demer_mean_prop_test()
            # env.demer_prop_test()
        env.expert_policy_info_gen()
        # city_env_dict['Heyuan-test'].expert_info = None
        # city_env_dict['Heyuan-test'].optimal_info = None
        eval_env.expert_policy_info_gen(env)
        break
        if args.alo_type == AlgorithmType.DFM:
            from transfer_learning.src.learn_deepfm import DeepFMLearner
            # params
            dfm_params = {
                "use_fm": True,
                "use_deep": True,
                "embedding_size": 32,
                "dropout_fm": [1.0, 1.0],
                "deep_layers": [128, 128],
                "dropout_deep": [0.5, 0.5, 0.5],
                "deep_layers_activation": tf.nn.relu,
                "epoch": int(1e6),
                "batch_size": 1024,
                "learning_rate": 0.001,
                "optimizer_type": "adam",
                "batch_norm": 1,
                "batch_norm_decay": 0.995,
                "l2_reg": 0.01,
                "verbose": True,
                "random_seed": args.seed,
                "loss_type": "mse",

            }
            dfml = DeepFMLearner(env=env, eval_env=eval_env, dfm_params=dfm_params, data_only=args.data_only)
            dfml.learn(total_timesteps=1000, log_interval=10)
            break
        np.random.seed(args.seed)
        if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
            if args.use_distribution_info:
                shift = features_dict[train_city_list[0]][1]
                scale = features_dict[train_city_list[0]][2]
                logger.info("shift-scale shape {}".format(shift.shape))
                vae_handler = vae_handle_initial(vae_tester, sess, 5, 'distribution_emb',
                                                 x_shape, max_category, raw_feature_len, days,
                                                 feature_name, condition_map_features, condition_index_bound_val,
                                                 remove_min_index,
                                                 categorical_index, log_norm_index, normal_index, is_min_vale_index,
                                                 static_feature_index, coupon_index, driver_index, shift=shift,
                                                 scale=scale, init_ops=False)

                vae_handler.load_from_checkpoint(vae_checkpoint_path) # TODO 增加load预测测试
                if args.vae_test:
                    vae_handler._init_embedding_predict_network()
                    vae_handler.embedding_predictable_evaluation(features_dict)
                assert args.stable_dist_embd
                zs = []
                for city in train_city_list + test_city_list:
                    features = features_dict[city]
                    if args.hand_code_distribution:
                        z = np.concatenate([features[0].mean(axis=(0, 1)), features[0].std(axis=(0, 1))], axis=0)
                    else:
                        z = vae_handler.z_step(features[0][0])
                    city_env_dict[city].env_z_info = z
                    zs.append(z)
                tester.simple_plot('vae', np.array(zs), labels=train_city_list + test_city_list)
            policy = EnvAwarePolicy
            env_extractor_policy = EnvExtractorPolicy

            env_extractor_policy_kwargs = {
                "n_lstm": args.n_lstm,
                "layers": args.env_extractor_layers,
                "use_resnet": args.use_resnet,
                "layer_norm": args.env_layer_norm,
                "lstm_layer_norm": args.lstm_layer_norm,
                "init_scale_output": args.init_scale_output,
            }
            if args.env_relu_act:
                env_extractor_policy_kwargs['act_fun'] = tf.nn.relu
            if args.env_out_relu_act:
                env_extractor_policy_kwargs['ent_out_act'] = tf.nn.leaky_relu
            else:
                env_extractor_policy_kwargs['ent_out_act'] = tf.nn.tanh

            policy_kwargs = {
                "stop_critic_gradient": args.stop_critic_gradient,
                "lstm_critic": args.lstm_critic,
                "n_lstm": args.n_lstm,
                "redun_distri_info": args.redun_distri,

            }
        else:
            policy = PpoMlp
            env_extractor_policy = None
            vae_handler = None
            env_extractor_policy_kwargs = None

            policy_kwargs = {}
        policy_kwargs["layers"] = args.policy_layers
        policy_kwargs["learn_var"] = args.learn_var
        policy_kwargs["res_net"] = args.res_net
        policy_kwargs["act_fun"] = ActFun.gen_act(args.policy_act)
        policy_kwargs["layer_norm"] = args.layer_norm
        policy_kwargs["init_scale"] = args.policy_init_scale

        log_interval = int(args.num_timesteps / 200 / 12 / args.batch_timesteps)

        from stable_baselines.common.schedules import LinearSchedule
        p_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.p_learning_rate * args.lr_remain_ratio, initial_p=args.p_learning_rate).value
        v_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.v_learning_rate * args.lr_remain_ratio, initial_p=args.v_learning_rate).value
        p_env_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.p_env_learning_rate * args.lr_remain_ratio, initial_p=args.p_env_learning_rate).value
        if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
            model = ppo2_env_aware(sess=sess, policy=policy, env_extractor_policy=env_extractor_policy,
                                   distribution_embedding=vae_handler, env=env, eval_env=eval_env,
                                   n_steps=args.samples_per_driver, nminibatches=1, lam=args.lam, gamma=args.gamma, noptepochs=args.noptepochs,
                                   name='ppo_model', ent_coef=args.ent_coef,
                                   v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn, p_env_learning_rate=p_env_lr_fn,
                                   cliprange=args.cliprange, verbose=1,
                                   full_tensorboard_log=args.full_tensorboard_log, policy_kwargs=policy_kwargs,
                                   env_extractor_policy_kwargs=env_extractor_policy_kwargs,
                                   consistent_ecoff=args.consistent_ecoff,
                                   v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm,
                                   use_lda_loss=args.use_lda_loss, driver_cluster_days=args.driver_cluster_days,
                                   lda_max_grad_norm=args.lda_max_grad_norm, constraint_lda=args.constraint_lda,
                                   log_lda=args.log_lda, normalize_returns=args.normalize_returns,
                                   hand_code_distribution=args.hand_code_distribution,
                                   env_params_size=args.env_params_size,
                                   stable_dist_embd=args.stable_dist_embd, just_scale=args.just_scale,
                                   normalize_rew=args.normalize_rew, scaled_lda=args.scaled_lda,
                                   cliped_lda=args.cliped_lda, use_cur_obs=args.use_cur_obs,
                                   l2_reg_pi=args.l2_reg_pi, l2_reg_env=args.l2_reg_env, l2_reg_v=args.l2_reg_v,
                                   merge_city_samples=args.merge_city_samples,
                                   lstm_train_freq=args.lstm_train_freq,
                                   rms_opt=args.rms_opt, remove_done=args.remove_done,
                                   soft_update_freq=args.soft_update_freq, log_interval=log_interval,
                                   square_lda=args.square_lda, tau=args.tau)
        else:
            if args.alo_type == AlgorithmType.PPO:
                model = ppo2_vanilla(sess=sess, policy=policy, env=env, eval_env=eval_env,
                                     n_steps=args.samples_per_driver, nminibatches=1, lam=args.lam, gamma=args.gamma, noptepochs=args.noptepochs,
                                     name='ppo_model', ent_coef=args.ent_coef,
                                       v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn,
                                     cliprange=args.cliprange, verbose=1,
                                     full_tensorboard_log=args.full_tensorboard_log, policy_kwargs=policy_kwargs,
                                     v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm,
                                     normalize_returns=args.normalize_returns,
                                     keep_dids_times=args.keep_dids_times, just_scale=args.just_scale,
                                     normalize_rew=args.normalize_rew, l2_reg=args.l2_reg_pi,
                                     l2_reg_v=args.l2_reg_v, merge_city_samples=args.merge_city_samples, log_interval=log_interval)
            elif args.alo_type == AlgorithmType.TRPO:
                model = trpo_vanilla(sess=sess, policy=policy, env=env, eval_env=eval_env, lam=args.lam,
                                     gamma=args.gamma, policy_kwargs=policy_kwargs, merge_city_samples=args.merge_city_samples,
                                     n_steps=args.samples_per_driver, entcoeff=args.ent_coef, verbose=1)
            else:
                raise NotImplementedError
        import tensorflow as tf
        sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)))

        tester.new_saver(model.name)
        if args.load_date is not '':
            tester.load_checkpoint(target_prefix_name='ppo_model/', current_name='ppo_model', sess=sess)
        tester.feed_hyper_params_to_tb()
        tester.print_large_memory_variable()

        model.learn(total_timesteps=args.num_timesteps, log_interval=log_interval)

