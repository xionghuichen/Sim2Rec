import gym
import sys
sys.path.append('../../')
import argparse
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds, tf_util as U
from lts.src.private_config import *
from common.tester import tester
from common.env_base import BaseDriverEnv
from transfer_learning.src.driver_simenv_disc import DriverEnv, MultiCityDriverEnv
from common.mlp_policy import MlpPolicy as DemerMlp
import numpy as np
from transfer_learning.src.policies import EnvExtractorPolicy, EnvAwarePolicy
from ppo2.policies import MlpPolicy as PpoMlp
from lts.src.lts_van_ppo import PPO2 as ppo2_vanilla
from lts.src.lts_env_aware_ppo import PPO2 as ppo2_env_aware
from trpo_mpi.trpo_mpi import TRPO as trpo_vanilla
import tensorflow as tf
from transfer_learning.src.func import *
from common.config import *
from common import logger
from lts.src.multi_user_env import make
from lts.src.recsim_gym import MultiDomainGymEnv
from lts.src.config import *
from lts.src.private_config import *


def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    parser.add_argument('--info', help='environment ID', default='')
    # parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--env_id', help='environment ID', default='Driver-vX-DEMER')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add track hyperparameter.
    return args

if __name__ == '__main__':
    args = argsparser()
    set_global_seeds(args.seed)
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    tester.clear_record_param()
    tester.add_record_param(['info'])
    sess = U.make_session(num_cpu=16).__enter__()
    def task_name_gen():
        task_name = '-'.join(['data_collect'])
        return task_name
    tester.configure(task_name_gen(), 'run_lts', record_date=args.load_date,
                     add_record_to_pkl=False, root=args.log_root, max_to_keep=1)
    vae_tester = tester.load_tester(vae_checkpoint_date, prefix_dir='all-norm-hier_ae',
                                    log_root=args.log_root + '../vae/')