from multiprocessing.sharedctypes import Value
import gym
import sys
sys.path.append('../../')
sys.path.append('../..')
import argparse
import xgboost as xgb
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds, tf_util as U
from lts.src.private_config import *
from common.tester import tester
import numpy as np
import json

from lts.src.policies import MlpPolicy as PpoMlp
from lts.src.lts_van_ppo import PPO2 as ppo2_vanilla
import tensorflow as tf
from transfer_learning.src.func import *
from common.config import *
from common import logger
from lts.src.multi_product_env import make_dh
from lts.src.dunnhumby_gym import MultiDomainGymEnv
from vae_lts.src.vae_handler import VAE
from transfer_learning.src.policies import ActorCriticPolicy
from transfer_learning.src.policies import EnvExtractorPolicy, EnvAwarePolicy
from stable_baselines.common.schedules import LinearSchedule
from lts.src.new_lts_env_aware_ppo import PPO2 as ppo2_env_aware
from lts.src.policies import LstmPolicy as policy
from stable_baselines.common.schedules import LinearSchedule


def task_name_gen():
    task_name = '-'.join(['v0'])
    return task_name

def argsparser():
    parser = argparse.ArgumentParser("Train pricing policy in simulator")
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--info', help='environment ID', default='')

    # tester
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    parser.add_argument('--load_date', type=str, default='')
    parser.add_argument('--load_sub_proj', type=str, default='transfer_lts')
    boolean_flag(parser, 'save_checkpoint', default=False)

    # learning configuration
    parser.add_argument('--noptepochs', type=int, default=3)
    boolean_flag(parser, 'norm_obs', default=False)
    boolean_flag(parser, 'simp_state', default=True)
    boolean_flag(parser, 'remove_done', default=True)
    boolean_flag(parser, 'rms_opt', default=False)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lstm_train_freq', type=int, default=1)
    parser.add_argument('--soft_update_freq', type=int, default=1)

    # setting of value and policy network training
    parser.add_argument('--v_grad_norm', type=float, default=0.5)
    parser.add_argument('--p_grad_norm', type=float, default=0.5)
    parser.add_argument('--lda_max_grad_norm', type=float, default=0.1)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=2e8) # need to be reduced for dunnhumby
    parser.add_argument('--batch_timesteps', help='number of timesteps per batch', type=int, default=30000)
    parser.add_argument('--keep_dids_times', help='number of timesteps per batch', type=int, default=1)

    # alg info
    parser.add_argument('--trans_type', help='environment ID', default=TransType.DIRECT_TRANS) #ENV_AWARE_VAN
    parser.add_argument('--alo_type', help='algorithm type', default=AlgorithmType.PPO)

    #policy network parameters
    parser.add_argument('--policy_layers', type=int, default=[128, 128, 128, 32], nargs='+')
    parser.add_argument('--post_policy_layers', type=int, default=[128, 64], nargs='+')
    parser.add_argument('--policy_act', type=str, default=ActFun.LEAKY_RELU)
    parser.add_argument('--ent_coef', type=float, default=0.02)
    parser.add_argument('--v_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_env_learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_rescale', type=float, default=0.5)
    boolean_flag(parser, 'redun_distri', default=True)
    boolean_flag(parser, 'oc', default=False)
    boolean_flag(parser, 'simple_lstm', default=True)
    boolean_flag(parser, 'plt_res', default=False)
    boolean_flag(parser, 'post_layer', default=False)
    parser.add_argument('--policy_init_scale', type=float, default=0.5)
    parser.add_argument('--lr_remain_ratio', type=float, default=0.01)
    parser.add_argument('--l2_reg_pi', type=float, default=1e-6)
    parser.add_argument('--l2_reg_v', type=float, default=1e-6)
    boolean_flag(parser, 'learn_var', default=False)
    boolean_flag(parser, 'normalize_rew', default=True)
    boolean_flag(parser, 'hand_code_distribution', default=False)
    boolean_flag(parser, 'just_scale', default=True)
    boolean_flag(parser, 'res_net', default=False)
    boolean_flag(parser, 'layer_norm', default=False)

    # train and test environments preparation
    parser.add_argument('--train_state_list', type=str, default=['KY']) # 'OH', 'TX'
    parser.add_argument('--test_state_list', type=str, default=['IN']) # 'IN', 'KY'
    parser.add_argument('--direct_trans_state_list', type=str, default=['OH'])

    # env extractor network parameters
    parser.add_argument('--l2_reg_env', type=float, default=1e-6)
    parser.add_argument('--product_cluster_days', type=int, default=4)
    parser.add_argument('--consistent_ecoff', type=float, default=0.8)
    parser.add_argument('--samples_per_product', type=int, default=4) # need to be customized for dunnhumby env
    parser.add_argument('--tau', type=int, default=0.001)
    parser.add_argument('--init_scale_output', type=float, default=0.5)
    parser.add_argument('--random_range', type=float, default=0.1)
    parser.add_argument('--env_extractor_layers', type=int, default=[64, 64], nargs='+')
    parser.add_argument('--n_lstm', type=int, default=64)
    boolean_flag(parser, 'use_lda_loss', default=False)
    boolean_flag(parser, 'pi_lda_loss', default=False)
    boolean_flag(parser, 'no_share_layer', default=True)
    boolean_flag(parser, 'use_resnet', default=False)
    boolean_flag(parser, 'stop_critic_gradient', default=False)
    boolean_flag(parser, 'dropout', default=False)
    boolean_flag(parser, 'log_lda', default=False)
    boolean_flag(parser, 'scaled_lda', default=True)
    boolean_flag(parser, 'cliped_lda', default=False)
    boolean_flag(parser, 'square_lda', default=True)
    parser.add_argument('--constraint_lda', type=str, default=ConstraintLDA.Proj2CGC)
    boolean_flag(parser, 'env_relu_act', default=True)
    boolean_flag(parser, 'env_out_relu_act', default=False)
    boolean_flag(parser, 'lstm_layer_norm', default=False)
    boolean_flag(parser, 'env_layer_norm', default=False)
    boolean_flag(parser, 'lstm_critic', default=True)
    boolean_flag(parser, 'lstm_policy', default=False)

    #distribution embedding parameters
    boolean_flag(parser, 'stable_dist_embd', default=True)
    boolean_flag(parser, 'vae_test', default=False)

    args = parser.parse_args()
    args.v_learning_rate *= args.lr_rescale
    args.p_learning_rate *= args.lr_rescale
    args.p_env_learning_rate *= args.lr_rescale

    if args.oc: #oracle critic
        args.no_share_layer = True
    if args.trans_type == TransType.DIRECT or args.trans_type == TransType.DIRECT_TRANS or args.trans_type == TransType.DIRECT_TRANS_CITY or args.trans_type == TransType.UNIVERSAL:
        args.ent_coef = 1e-5
    else:
        args.ent_coef = 0.02

    assert args.cliped_lda + args.square_lda <= 1
    # add track hyperparameters

    return args


if __name__ == '__main__':
    args = argsparser()

    with open('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/store_state.json') as f:
        store_state = json.load(f)

    #set_global_seeds(args.seed)
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.clear_record_param()
    tester.add_record_param(['info', 'seed', 'trans_type', 'test_state_list'])
    sess = U.make_session(num_cpu=16).__enter__()

    tester.configure(task_name_gen(), 'run_lts', record_date=args.load_date, add_record_to_pkl=False, root=args.log_root, max_to_keep=1)
    tester.print_args()
    vae_handler=None
    env_dict = {}
    eval_env_dict = {}

    if args.trans_type == TransType.ENV_AWARE:
        # load tester
        vae_checkpoint_date = vae_checkpoint_dates[3]
        vae_tester = tester.load_tester(vae_checkpoint_date, prefix_dir='lts_ae', log_root=args.log_root + vae_root)

        # load vae
        dir, load_model_path = vae_tester.load_checkpoint_from_date(vae_checkpoint_date, prefix_dir='lts_ae', log_root=args.log_root + vae_root)
        vae_checkpoint_path = dir + '/' + load_model_path
        vae_handler = VAE(
            vae_tester.hyper_param['hid_size'],
            vae_tester.hyper_param['hid_layers'],
            vae_tester.hyper_param['layer_norm'],
            vae_tester.hyper_param['constant_z_scale'],
            vae_tester.hyper_param['lr'],
            vae_tester.hyper_param['res_struc'],
            [1],
            vae_tester.hyper_param['z_size'],
            np.array([0]),
            sess,
            'vae',
            vae_tester.hyper_param['l2_reg_coeff']
        )
        vae_handler.load_from_checkpoint(vae_checkpoint_path)
   
    # define env
    ## train env
    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.UNIVERSAL:
        for store, state in store_state.items():
            if state in args.train_state_list:
                env = make_dh(str(store))
                env_dict[str(store)] = env
    elif args.trans_type == TransType.ENV_AWARE_DIRECT or args.trans_type == TransType.DIRECT:
        for store, state in store_state.items():
            if state in args.test_state_list:
                env = make_dh(str(store)) 
                env_dict[str(store)] = env
    elif args.trans_type == TransType.DIRECT_TRANS:
        for store, state in store_state.items():
            if state in args.direct_trans_state_list:
                env = make_dh(str(store))
                env_dict[str(store)] = env
    else:
        raise NotImplementedError

    print('env_dict', env_dict)
    print('domain list', list(env_dict.keys()))
 
    # one store, one domain
    env = MultiDomainGymEnv(env_dict=env_dict, domain_list=list(env_dict.keys()), num_domain=len(list(env_dict.keys())))
    all_domain_list = list(env_dict.keys())
    
    ## test env
    for store, state in store_state.items():
        if state in args.test_state_list:
            eval_env = make_dh(str(store))
            eval_env_dict[str(store) + '-test'] = eval_env
            env_dict[str(store) + '-test'] = eval_env
            all_domain_list.append(str(store) + '-test')

    eval_env = MultiDomainGymEnv(env_dict=eval_env_dict, domain_list=list(eval_env_dict.keys()), num_domain=len(list(eval_env_dict.keys())))
    print('full domain list', list(env_dict.keys()))


    # make policy parameters
    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
        policy = EnvAwarePolicy
        assert issubclass(policy, ActorCriticPolicy)
        env_extractor_policy = EnvExtractorPolicy
        env_extractor_policy_kwargs = {
            "n_lstm": args.n_lstm,
            "layers": args.env_extractor_layers,
            "use_resent": args.use_resnet,
            "layer_norm": args.env_layer_norm,
            "lstm_layer_norm": args.lstm_layer_norm,
            "init_scale_output": args.init_scale_output,
            "dropout": args.dropout
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
            "redun_distri_info":args.redun_distri,
            "dropout": args.dropout,
            "lstm_policy": args.lstm_policy
        }
    else:
        policy = PpoMlp
        env_extractor_policy = None
        env_extractor_policy_kwargs = None

        policy_kwargs = {}
    
    policy_kwargs["layers"] = args.policy_layers
    policy_kwargs["learn_var"] = args.learn_var
    policy_kwargs["res_net"] = args.res_net
    policy_kwargs["act_fun"] = ActFun.gen_act(args.policy_act)
    policy_kwargs["layer_norm"] = args.layer_norm 
    policy_kwargs["init_scale"] = args.policy_init_scale
    log_interval = int(args.num_timesteps / 100 / 12 / args.batch_timesteps)

    p_lr_fn = LinearSchedule(schedule_timesteps=1.0, final_p=args.p_learning_rate * args.lr_remain_ratio, initial_p=args.p_learning_rate).value
    v_lr_fn = LinearSchedule(schedule_timesteps=1.0, final_p=args.v_learning_rate * args.lr_remain_ratio, initial_p=args.v_learning_rate).value
    p_env_lr_fn = LinearSchedule(schedule_timesteps=1.0, final_p=args.p_env_learning_rate * args.lr_remain_ratio, initial_p=args.p_env_learning_rate).value

    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
        if args.simple_lstm:
            policy_kwargs = {}
            policy_kwargs["n_lstm"] = args.n_lstm
            policy_kwargs["act_fun"] = ActFun.gen_act(args.policy_act)
            policy_kwargs["layer_norm"] = args.layer_norm
            policy_kwargs["lstm_layer_norm"] = args.lstm_layer_norm
            policy_kwargs["dropout"] = args.dropout
            policy_kwargs["no_share_layer"] = args.no_share_layer
            policy_kwargs["oc"] = args.oc
            if args.post_layer:
                policy_kwargs["layers"] = args.env_extractor_layers
                policy_kwargs["post_policy_layers"] = args.policy_layers
            else:
                policy_kwargs["layers"] = args.policy_layers
                policy_kwargs["post_policy_layers"] = args.post_policy_layers
                policy_kwargs["redun_info"] = args.redun_distri

            if args.stop_critic_gradient:
                policy_kwargs["redun_info"] = True
                policy_kwargs["stop_critic_gradient"] = args.stop_critic_gradient
            policy_kwargs['feature_extraction'] = "mlp"
            model = ppo2_env_aware(sess=sess, policy=policy, env_extractor_policy=env_extractor_policy, vae_handler=vae_handler, env=env, eval_env=eval_env, n_steps=args.samples_per_product, nminibatches=1, lam=args.lam, gamma=args.gamma,
                                   noptepochs=args.noptepochs, random_range=args.random_range, name='ppo_model', ent_coef=args.ent_coef, v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn, p_env_learning_rate=p_env_lr_fn, cliprange=args.cliprange, verbose=1,
                                   full_tensorboard_log=False, policy_kwargs=policy_kwargs, env_extractor_policy_kwargs=env_extractor_policy_kwargs, consistent_ecoff=args.consistent_ecoff, v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm, 
                                   use_lda_loss=args.use_lda_loss, pi_lda_loss=args.pi_lda_loss, driver_cluster_days=args.product_cluster_days, lda_max_grad_norm=args.lda_max_grad_norm, constraint_lda=args.constraint_lda, log_lda=args.log_lda, hand_code_distribution=args.hand_code_distribution,
                                   env_params_size=args.n_lstm, stable_dist_embd=args.stable_dist_embd, just_scale=args.just_scale, normalize_rew=args.normalize_rew, scaled_lda=args.scaled_lda, cliped_lda=args.cliped_lda, l2_reg_pi=args.l2_reg_pi, l2_reg_env=args.l2_reg_env, l2_reg_v=args.l2_reg_v,
                                   lstm_train_freq=args.lstm_train_freq, rms_opt=args.rms_opt, remove_done=args.remove_done, soft_update_freq=args.soft_update_freq, log_interval=log_interval, square_lda=args.square_lda, tau=args.tau, all_domain_list=all_domain_list)




            model = ppo2_env_aware()

        else:
            model = ppo2_env_aware(sess=sess, policy=policy, env_extractor_policy=env_extractor_policy,
                                   distribution_embedding=vae_handler, env=env, eval_env=eval_env, n_steps=args.samples_per_product, nminibatches=1, lam=args.lam,
                                   gamma=args.gamma, noptepochs=args.noptepochs, name='ppo_model', ent_coef=args.ent_coef, v_learing_rate=v_lr_fn, p_learning_rate=p_lr_fn, p_env_learning_rate=p_env_lr_fn,
                                   cliprange=args.cliprange, verbose=1, full_tensorboard_log=False, policy_kwargs=policy_kwargs, env_extractor_policy_kwargs=env_extractor_policy_kwargs, consistent_ecoff=args.consistent_ecoff,
                                   v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm, use_lda_loss=args.use_lda_loss, driver_cluster_days=args.product_cluster_days, lda_max_grad_norm=args.lda_max_grad_norm, constraint_lda=args.constraint_lda,
                                   log_lda=args.log_lda, hand_code_distribution=args.hand_code_distribution, env_params_size=args.n_lstm, stable_dist_embd=args.stable_dist_embd, just_scale=args.just_scale, normalize_rew=args.normalize_rew,
                                   scaled_lda=args.scaled_lda, cliped_lda=args.cliped_lda, l2_reg_pi=args.l2_reg_pi, l2_reg_env=args.l2_reg_env, l2_reg_v=args.l2_reg_v, lstm_train_freq=args.lstm_train_freq, rms_opt=args.rms_opt, remove_done=args.remove_done,
                                   soft_update_freq=args.soft_update_freq, log_interval=log_interval, square_lda=args.square_lda, tau=args.tau, all_domain_list=all_domain_list)

    else:
        model = ppo2_vanilla(sess=sess, policy=policy, env=env, eval_env=eval_env, 
                             n_steps=args.samples_per_product, nminibatches=1, lam=args.lam, gamma=args.gamma,
                             noptepochs=args.noptepochs, name='ppo_model', ent_coef=args.ent_coef, v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn,
                             cliprange=args.cliprange, verbose=1, full_tensorboard_log=False, policy_kwargs=policy_kwargs, v_grad_norm=args.v_grad_norm,
                             p_grad_norm=args.p_grad_norm, keep_dids_times=args.keep_dids_times, just_scale=args.just_scale, normalize_rew=args.normalize_rew,
                             l2_reg=args.l2_reg_pi, l2_reg_v=args.l2_reg_v, log_interval=log_interval, all_domain_list=all_domain_list)
    
    sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)))

    # vae performance test.
    if args.trans_type == TransType.ENV_AWARE:
        code, recons_data = vae_handler.reconstruct_samples()

    tester.new_saver(model.name)
    if args.load_date is not '':
        tester.load_checkpoint(target_prefix_name='ppo_model/', current_name='ppo_model', sess=sess)


    tester.feed_hyper_params_to_tb()
    tester.print_large_memory_variable()

    model.learn(total_timesteps=args.num_timesteps, log_interval=log_interval)
    