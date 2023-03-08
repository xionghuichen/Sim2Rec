import gym
import sys
sys.path.append('../../')
sys.path.append('../../')
import argparse
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds, tf_util as U
from lts.src.private_config import *
from common.tester import tester
import numpy as np

from lts.src.policies import MlpPolicy as PpoMlp
from lts.src.lts_van_ppo import PPO2 as ppo2_vanilla
#from lts.src.lts_env_aware_ppo import PPO2 as ppo2_env_aware
import tensorflow as tf
from transfer_learning.src.func import *
from common.config import *
from common import logger
from lts.src.multi_user_env import make
from lts.src.recsim_gym import MultiDomainGymEnv
from lts.src.config import *

def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--info', help='environment ID', default='')

    # tester
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    parser.add_argument('--load_date', type=str, default='')
    parser.add_argument('--load_sub_proj', type=str, default='transfer_lts')
    boolean_flag(parser, 'save_checkpoint', default=False)
    # learning configuration
    parser.add_argument('--noptepochs', type=int, default=3)  # modified
    boolean_flag(parser, 'norm_obs', default=False)  # hurt perf, check bug.
    boolean_flag(parser, 'simp_state', default=True)
    boolean_flag(parser, 'remove_done', default=True)
    boolean_flag(parser, 'budget_constraint', default=True)
    parser.add_argument('--budget_expand', type=float, default=1.5)
    parser.add_argument('--trans_level', type=float, default=4.0)
    boolean_flag(parser, 'UE_penalty', default=True)
    boolean_flag(parser, 'gmv_rescale', default=True)
    boolean_flag(parser, 'merge_samples', default=True)
    boolean_flag(parser, 'rms_opt', default=False)
    parser.add_argument('--lam', type=float, default=0.95)
    boolean_flag(parser, 'log_sample', default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lstm_train_freq', type=int, default=1)

    parser.add_argument('--soft_update_freq', type=int, default=1)
    # v: 200, p: 20 seems to be better
    parser.add_argument('--v_grad_norm', type=float, default=0.5)
    parser.add_argument('--p_grad_norm', type=float, default=0.5)
    parser.add_argument('--lda_max_grad_norm', type=float, default=0.1)
    parser.add_argument('--cliprange', type=float, default=0.2)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=20e8)
    parser.add_argument('--batch_timesteps', help='number of timesteps per batch', type=int, default=30000)
    # parser.add_argument('--batch_size', help='number of timesteps per batch', type=int, default=4096)
    parser.add_argument('--keep_dids_times', help='number of timesteps per batch', type=int, default=1)
    # alg info
    parser.add_argument('--trans_type', help='environment ID', default=TransType.ENV_AWARE)
    parser.add_argument('--alo_type', help='environment ID', default=AlgorithmType.PPO)
    parser.add_argument('--time_budget', type=int, default=140)
    parser.add_argument('--sim_noise_level', type=int, default=0)
    # policy network parameters
    parser.add_argument('--policy_layers', type=int, default=[128, 128, 128, 32], nargs='+') #[512, 128]
    parser.add_argument('--post_policy_layers', type=int, default=[128, 64], nargs='+') #[512, 128]
    parser.add_argument('--policy_act', type=str, default=ActFun.LEAKY_RELU)
    parser.add_argument('--ent_coef', type=float, default=0.02)
    parser.add_argument('--v_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_learning_rate', type=float, default=1e-4)
    parser.add_argument('--p_env_learning_rate', type=float, default=1e-4)
    parser.add_argument('--choc_mean', type=float, default=5)
    parser.add_argument('--test_choc_mean', type=float, default=-1)
    parser.add_argument('--lr_rescale', type=float, default=0.5)
    parser.add_argument('--kale_mean', type=float, default=4.0)
    boolean_flag(parser, 'redun_distri', default=True)
    boolean_flag(parser, 'given_ep', default=False)
    boolean_flag(parser, 'oc', default=False)
    boolean_flag(parser, 'simple_lstm', default=True)
    boolean_flag(parser, 'plt_res', default=False)
    boolean_flag(parser, 'post_layer', default=False)

    parser.add_argument('--policy_init_scale', type=float, default=0.5)
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
    boolean_flag(parser, 'layer_norm', default=False)
    # env extractor network parameters
    parser.add_argument('--l2_reg_env', type=float, default=1e-6)
    parser.add_argument('--driver_cluster_days', type=int, default=3)  # driver_cluster_days = 7 hurt performance.
    parser.add_argument('--env_params_size', type=int, default=64)
    parser.add_argument('--consistent_ecoff', type=float, default=0.8)
    parser.add_argument('--samples_per_driver', type=int, default=60)  # 30 可以提升所有方法的性能， 测试更困难的环境
    parser.add_argument('--tau', type=int, default=0.001)
    parser.add_argument('--init_scale_output', type=float, default=0.5)
    parser.add_argument('--random_range', type=float, default=0.1)  # 大于0.1 的效果都变差了
    parser.add_argument('--env_extractor_layers', type=int, default=[64, 64], nargs='+')
    parser.add_argument('--n_lstm', type=int, default=64)
    parser.add_argument('--cgc_type', type=int, default=10)
    parser.add_argument('--gp_range', type=int, default=8)
    parser.add_argument('--std_level', type=int, default=3)
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
    boolean_flag(parser, 'use_cur_obs', default=True)
    boolean_flag(parser, 'env_relu_act', default=True)
    boolean_flag(parser, 'env_out_relu_act', default=False)
    boolean_flag(parser, 'lstm_layer_norm', default=False)
    boolean_flag(parser, 'env_layer_norm', default=False)
    boolean_flag(parser, 'lstm_critic', default=True)
    boolean_flag(parser, 'lstm_policy', default=False)

    # distribution embedding parameters
    boolean_flag(parser, 'stable_dist_embd', default=True)
    boolean_flag(parser, 'use_distribution_info', default=False)
    boolean_flag(parser, 'vae_test', default=False)
    args = parser.parse_args()
    args.v_learning_rate *= args.lr_rescale
    args.p_learning_rate *= args.lr_rescale
    args.p_env_learning_rate *= args.lr_rescale
    # args.lr_remain_ratio = args.lr_remain_ratio * 5e8 / args.num_timesteps
    if args.oc: # oracle critic
        args.given_ep = True
        args.no_share_layer = True
    if args.trans_type == TransType.DIRECT or args.trans_type == TransType.DIRECT_TRANS or args.trans_type == TransType.DIRECT_TRANS_CITY\
            or args.trans_type == TransType.UNIVERSAL:
        args.ent_coef = 1e-5
    else:
        args.ent_coef = 0.02

    assert args.cliped_lda + args.square_lda <= 1
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add track hyperparameter.
    return args

# Custom MLP policy of three layers of size 128 each

if __name__ == '__main__':
    args = argsparser()
    set_global_seeds(args.seed)
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.clear_record_param()
    tester.add_record_param(['info','seed',
                             'trans_type', 'given_ep', "test_choc_mean", "std_level", "trans_level", "sim_noise_level",
                             # "batch_timesteps",
                             # "num_timesteps",
                             # "lr_rescale",
                              # "scaled_lda", "random_range", "use_lda_loss", "pi_lda_loss",
                             # "n_lstm", "ent_coef",
                             # "samples_per_driver",
                             # "no_share_layer", "lr_remain_ratio", "merge_samples",
                             # 'cliped_lda',  'square_lda',  'use_lda_loss', "lstm_policy","scaled_lda"
                             ])
    sess = U.make_session(num_cpu=16).__enter__()
    def task_name_gen():
        task_name = '-'.join(['v0'])
        return task_name
    tester.configure(task_name_gen(), 'run_lts', record_date=args.load_date,
                     add_record_to_pkl=False, root=args.log_root,
                     max_to_keep=1)
    tester.print_args()
    # make_env
    # preprocess dataset.
    CHOC_BONUS_RANGE[0] = 14 - args.gp_range
    CHOC_BONUS_RANGE[1] = 14 + args.gp_range
    vae_handler = None
    env_dict = {}
    if args.test_choc_mean == -1:
        test_choc_mean = (CHOC_BONUS_RANGE[1] - CHOC_BONUS_RANGE[0]) / 2 + CHOC_BONUS_RANGE[0]
    else:
        test_choc_mean = args.test_choc_mean
    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.UNIVERSAL:
        given_gp = True
    else:
        given_gp = False
    if args.trans_type == TransType.ENV_AWARE:
        # [vae] load vae op.
        from vae_lts.src.vae_handler import VAE
        # load tester
        vae_checkpoint_date = vae_checkpoint_dates[args.std_level]
        vae_tester = tester.load_tester(vae_checkpoint_date, prefix_dir='lts_ae',
                                        log_root=args.log_root + vae_root)
        # load vae
        dir, load_model_path = vae_tester.load_checkpoint_from_date(vae_checkpoint_date,
                                                                    prefix_dir='lts_ae',
                                                                    log_root=args.log_root + vae_root)
        vae_checkpoint_path = dir + '/' + load_model_path
        vae_handler = VAE(vae_tester.hyper_param['hid_size'],
                  vae_tester.hyper_param['hid_layers'],
                  vae_tester.hyper_param['layer_norm'],
                  vae_tester.hyper_param['constant_z_scale'], vae_tester.hyper_param['lr'],
                  vae_tester.hyper_param['res_struc'], [1], vae_tester.hyper_param['z_size'],
                  np.array([0]),  sess, 'vae', vae_tester.hyper_param['l2_reg_coeff'])
        assert args.gp_range == vae_tester.hyper_param['gp_range']
        assert args.trans_level == vae_tester.hyper_param['trans_level']
        vae_handler.load_from_checkpoint(vae_checkpoint_path)

    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.UNIVERSAL:
        if args.merge_samples:
            num_users = int(args.batch_timesteps / args.samples_per_driver / ((args.gp_range - args.trans_level) * 2))
        else:
            num_users = int(args.batch_timesteps / args.samples_per_driver)
        print('CHOC_BONUS_RANGE', CHOC_BONUS_RANGE)
        for i in range(CHOC_BONUS_RANGE[0], CHOC_BONUS_RANGE[1]):
            if np.abs(test_choc_mean - i) <= args.trans_level:
                continue
            env = make(num_users=num_users, time_budget=args.time_budget, domain_name=str(i), cgc_type=args.cgc_type,
                       given_ep=args.given_ep, choc_mean=i, kale_mean=args.kale_mean, given_gp=given_gp,
                       log_sample=args.log_sample, std_level=args.std_level, sim_noise_level=args.sim_noise_level)
            env_dict[str(i)] = env
    elif args.trans_type == TransType.ENV_AWARE_DIRECT or args.trans_type == TransType.DIRECT:
        num_users = int(args.batch_timesteps / args.samples_per_driver)
        env = make(num_users=num_users, time_budget=args.time_budget, domain_name=str(test_choc_mean), cgc_type=args.cgc_type,
                   given_ep=args.given_ep, choc_mean=test_choc_mean, kale_mean=args.kale_mean, given_gp=given_gp,
                   log_sample=args.log_sample, std_level=args.std_level, sim_noise_level=args.sim_noise_level)
        env_dict[str(test_choc_mean)] = env
    elif args.trans_type == TransType.DIRECT_TRANS:
        num_users = int(args.batch_timesteps / args.samples_per_driver)
        choc_mean = test_choc_mean - args.trans_level - 1
        env = make(num_users=num_users, time_budget=args.time_budget, domain_name=str(test_choc_mean),
                   cgc_type=args.cgc_type,
                   given_ep=args.given_ep, choc_mean=choc_mean, kale_mean=args.kale_mean,
                   given_gp=given_gp, log_sample=args.log_sample, std_level=args.std_level)
        env_dict[str(test_choc_mean)] = env
    else:
        print('args.trans_type == TransType.ENV_AWARE', args.trans_type == TransType.ENV_AWARE)
        raise NotImplementedError
    eval_env = make(num_users=750, time_budget=args.time_budget, domain_name=str(test_choc_mean) + '-test', cgc_type=args.cgc_type,
                    given_ep=args.given_ep, choc_mean=test_choc_mean, kale_mean=args.kale_mean,
                    given_gp=given_gp, log_sample=args.log_sample, std_level=args.std_level)
    print('env_dict', env_dict)
    print('domain_list', list(env_dict.keys()))
    env = MultiDomainGymEnv(env_dict=env_dict, domain_list=list(env_dict.keys()), num_domain=len(list(env_dict.keys())))
    all_domain_list = list(env_dict.keys())
    mask_len = env.selected_env.env._environment._user_model[0].observation_space().shape[0]
    if args.oc:
        assert mask_len > 0
    assert str(test_choc_mean) + '-test' not in all_domain_list
    all_domain_list.append(str(test_choc_mean) + '-test')
    env_dict[str(test_choc_mean) + '-test'] = eval_env
    print('full domain list', list(env_dict.keys()))

    # make policy parameters
    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
        from transfer_learning.src.policies import EnvExtractorPolicy, EnvAwarePolicy
        policy = EnvAwarePolicy
        env_extractor_policy = EnvExtractorPolicy
        env_extractor_policy_kwargs = {
            "n_lstm": args.n_lstm,
            "layers": args.env_extractor_layers,
            "use_resnet": args.use_resnet,
            "layer_norm": args.env_layer_norm,
            "lstm_layer_norm": args.lstm_layer_norm,
            "init_scale_output": args.init_scale_output,
            "dropout": args.dropout,

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
            "dropout": args.dropout,
            "lstm_policy": args.lstm_policy,

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

    from stable_baselines.common.schedules import LinearSchedule
    p_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.p_learning_rate * args.lr_remain_ratio,
                             initial_p=args.p_learning_rate).value
    v_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.v_learning_rate * args.lr_remain_ratio,
                             initial_p=args.v_learning_rate).value
    p_env_lr_fn = LinearSchedule(schedule_timesteps=1., final_p=args.p_env_learning_rate * args.lr_remain_ratio,
                                 initial_p=args.p_env_learning_rate).value
    if args.trans_type == TransType.ENV_AWARE or args.trans_type == TransType.ENV_AWARE_VAN or args.trans_type == TransType.ENV_AWARE_DIRECT:
        if args.simple_lstm:
            from lts.src.new_lts_env_aware_ppo import PPO2 as ppo2_env_aware
            from lts.src.policies import LstmPolicy as policy
            policy_kwargs = {}
            policy_kwargs["n_lstm"] = args.n_lstm
            policy_kwargs["act_fun"] = ActFun.gen_act(args.policy_act)
            policy_kwargs["layer_norm"] = args.layer_norm
            policy_kwargs["lstm_layer_norm"] = args.lstm_layer_norm
            policy_kwargs["dropout"] = args.dropout
            policy_kwargs["no_share_layer"] = args.no_share_layer
            policy_kwargs["mask_len"] = mask_len
            policy_kwargs["oc"] = args.oc
            if args.post_layer:
                policy_kwargs["layers"] = args.env_extractor_layers
                policy_kwargs["post_policy_layers"] = args.policy_layers
            else:
                policy_kwargs["layers"] = args.policy_layers
                policy_kwargs["post_policy_layers"] = args.post_policy_layers
                policy_kwargs['redun_info'] = args.redun_distri
            if args.stop_critic_gradient:
                policy_kwargs['redun_info'] = True
                policy_kwargs['stop_critic_gradient'] = args.stop_critic_gradient
            policy_kwargs['feature_extraction'] = "mlp"
            model = ppo2_env_aware(sess=sess, policy=policy, env_extractor_policy=env_extractor_policy,
                                   vae_handler=vae_handler, env=env, eval_env=eval_env,
                                   n_steps=args.samples_per_driver, nminibatches=1, lam=args.lam, gamma=args.gamma,
                                   noptepochs=args.noptepochs, random_range=args.random_range,
                                   name='ppo_model', ent_coef=args.ent_coef,
                                   v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn, p_env_learning_rate=p_env_lr_fn,
                                   cliprange=args.cliprange, verbose=1,
                                   full_tensorboard_log=False, policy_kwargs=policy_kwargs,
                                   env_extractor_policy_kwargs=env_extractor_policy_kwargs,
                                   consistent_ecoff=args.consistent_ecoff,
                                   v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm,
                                   use_lda_loss=args.use_lda_loss, pi_lda_loss=args.pi_lda_loss, driver_cluster_days=args.driver_cluster_days,
                                   lda_max_grad_norm=args.lda_max_grad_norm, constraint_lda=args.constraint_lda,
                                   log_lda=args.log_lda,
                                   hand_code_distribution=args.hand_code_distribution,
                                   env_params_size=args.n_lstm,
                                   stable_dist_embd=args.stable_dist_embd, just_scale=args.just_scale,
                                   normalize_rew=args.normalize_rew, scaled_lda=args.scaled_lda,
                                   cliped_lda=args.cliped_lda,
                                   l2_reg_pi=args.l2_reg_pi, l2_reg_env=args.l2_reg_env, l2_reg_v=args.l2_reg_v,
                                   lstm_train_freq=args.lstm_train_freq,
                                   rms_opt=args.rms_opt, remove_done=args.remove_done,
                                   soft_update_freq=args.soft_update_freq, log_interval=log_interval,
                                   square_lda=args.square_lda, tau=args.tau, all_domain_list=all_domain_list, merge_samples=args.merge_samples)
        else:
            model = ppo2_env_aware(sess=sess, policy=policy, env_extractor_policy=env_extractor_policy,
                                   distribution_embedding=vae_handler, env=env, eval_env=eval_env,
                                   n_steps=args.samples_per_driver, nminibatches=1, lam=args.lam, gamma=args.gamma,
                                   noptepochs=args.noptepochs,
                                   name='ppo_model', ent_coef=args.ent_coef,
                                   v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn, p_env_learning_rate=p_env_lr_fn,
                                   cliprange=args.cliprange, verbose=1,
                                   full_tensorboard_log=False, policy_kwargs=policy_kwargs,
                                   env_extractor_policy_kwargs=env_extractor_policy_kwargs,
                                   consistent_ecoff=args.consistent_ecoff,
                                   v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm,
                                   use_lda_loss=args.use_lda_loss, driver_cluster_days=args.driver_cluster_days,
                                   lda_max_grad_norm=args.lda_max_grad_norm, constraint_lda=args.constraint_lda,
                                   log_lda=args.log_lda,
                                   hand_code_distribution=args.hand_code_distribution,
                                   env_params_size=args.n_lstm,
                                   stable_dist_embd=args.stable_dist_embd, just_scale=args.just_scale,
                                   normalize_rew=args.normalize_rew, scaled_lda=args.scaled_lda,
                                   cliped_lda=args.cliped_lda,
                                   l2_reg_pi=args.l2_reg_pi, l2_reg_env=args.l2_reg_env, l2_reg_v=args.l2_reg_v,
                                   lstm_train_freq=args.lstm_train_freq,
                                   rms_opt=args.rms_opt, remove_done=args.remove_done,
                                   soft_update_freq=args.soft_update_freq, log_interval=log_interval,
                                   square_lda=args.square_lda, tau=args.tau, all_domain_list=all_domain_list)
    else:
        model = ppo2_vanilla(sess=sess, policy=policy, env=env, eval_env=eval_env,
                             n_steps=args.samples_per_driver, nminibatches=1, lam=args.lam, gamma=args.gamma,
                             noptepochs=args.noptepochs,
                             name='ppo_model', ent_coef=args.ent_coef,
                             v_learning_rate=v_lr_fn, p_learning_rate=p_lr_fn,
                             cliprange=args.cliprange, verbose=1,
                             full_tensorboard_log=False, policy_kwargs=policy_kwargs,
                             v_grad_norm=args.v_grad_norm, p_grad_norm=args.p_grad_norm,
                             keep_dids_times=args.keep_dids_times, just_scale=args.just_scale,
                             normalize_rew=args.normalize_rew, l2_reg=args.l2_reg_pi,
                             l2_reg_v=args.l2_reg_v,
                             log_interval=log_interval, all_domain_list=all_domain_list,
                             merge_samples=args.merge_samples)
    sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)))
    # vae performance test.
    if args.trans_type == TransType.ENV_AWARE and args.test_choc_mean == -1:
        def compute_kl(real_mean, real_std, recons_data):
            real_logstd = np.log(real_std)
            recons_logstd = np.log(np.std(recons_data))
            recons_std = np.std(recons_data)
            recons_mean = np.mean(recons_data)
            kld = np.sum(recons_logstd - real_logstd + (np.square(real_std) + np.square(real_mean - recons_mean)) / (
                    2 * np.square(recons_std)) - 0.5, axis=-1)
            return kld
        test_gp_data = np.random.normal(test_choc_mean, 1, size=(num_users, 1))
        code, recons_data = vae_handler.reconstruct_samples(test_gp_data, num_users)
        real_std = args.std_level
        real_mean = test_choc_mean
        kld = compute_kl(real_mean, real_std, recons_data)
        logger.info("kld:{}".format(kld))
        logger.record_tabular("vae-test/kld", kld)
        #assert kld < 0.3, "kld is too large"
    tester.new_saver(model.name)
    if args.load_date is not '':
        tester.load_checkpoint(target_prefix_name='ppo_model/', current_name='ppo_model', sess=sess)
    tester.feed_hyper_params_to_tb()
    tester.print_large_memory_variable()

    model.learn(total_timesteps=args.num_timesteps, log_interval=log_interval)
