import argparse
import xgboost as xgb
import sys

sys.path.append('../../')
import json
import pandas as pd

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from common import logger
from common.utils import *
from common.config import *
# from vae.src.private_config import *
from common.tester import tester
from vae_lts.src.vae_handler import VAE
#from vae.vae_cls_fix import VAE

from lts.src.config import CHOC_BONUS_RANGE
from common.private_config import *
from common.tester import tester
#from sklearn.decomposition import PCA
from lts.src.dunnhumby_env import SingleStoreEnvironment
from lts.src.store_sales_model import StoreSalesModel

def compute_kl(real_mean, real_std, recons_data):
    real_logstd = np.log(real_std)
    recons_logstd = np.log(np.std(recons_data))
    recons_std = np.std(recons_data)
    recons_mean = np.mean(recons_data)
    kld = np.sum(recons_logstd - real_logstd + (np.square(real_std) + np.square(real_mean - recons_mean)) / (
                2 * np.square(recons_std)) - 0.5, axis=-1)
    return kld

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DEMER")
    #task parameters
    parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--task', help='environment ID', default='lts_ae')
    boolean_flag(parser, 'just_evaluation', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)

    #data load setting
    parser.add_argument('--train_state_list', type=str, default=['KY']) #['OH', 'TX']
    parser.add_argument('--test_state_list', type=str, default=['IN']) #['IN', 'KY']
    parser.add_argument('--direct_trans_state_list', type=str, default=['OH'])

    #vae parameters
    parser.add_argument('--z_size', type=int, default=3)
    parser.add_argument('--hid_layers', type=int, default=2)
    parser.add_argument('--l2_reg_coeff', type=float, default=0.1)
    parser.add_argument('--hid_size', type=int, default=512)
    parser.add_argument('--sample_product_num', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00002) #0.00002
    boolean_flag(parser, 'layer_norm', default=True)
    boolean_flag(parser, 'constant_z_scale', default=False)
    boolean_flag(parser, 'res_struc', default=False)

    #train parameters
    parser.add_argument('--epochs', type=int, default=10000)

    #test-obj parameters
    parser.add_argument('--log_root', type=str, default=LOG_ROOT)
    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)

    #add track hyperparameters
    tester.add_record_param(
        ['info', 'z_size', 'seed']
    )

    return args

def vae_data_info_preprocess(env):
    trajectory = None

    assert isinstance(env, SingleStoreEnvironment)
    trajectory = env.simulate_trajectory()

    return trajectory




if __name__ == '__main__':
    args = argsparser()

    merged_data = pd.read_csv('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/merged_data_norm_1022.csv')
    with open('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/store_state.json') as f:
        store_state = json.load(f)

    feature_list = ['FEATURE', 'DISPLAY', 'TPR_ONLY', 'PRICE', 'BASE_PRICE']

    store_list = [key for key, value in store_state.items()]
    train_store_list = []

    sess = U.make_session(num_cpu=16).__enter__()
    set_global_seeds(args.seed)
    task_name = tester.task_gen([args.task])
    tester.configure(task_name, 'run_ae', add_record_to_pkl=False, root=args.log_root,
                     max_to_keep=3)


    tester.print_args()  
    len_episode = 4
    num_product = 55
    dim_feature = 5 + 1
    env_dict = {}
    data_dict = {}
    test_data_dict = {}


    # preprocess dataset
    
    # train  

    for store, state in store_state.items():
        if state in args.train_state_list:
            print('store', store)
            train_store_list.append(store)
            data_dict[store] = merged_data[merged_data['STORE_ID'] == int(store)][feature_list].values.reshape(-1, 5)
    for store, state in store_state.items():
        if state in args.test_state_list:
            print('store', store)
            test_data_dict[store] = merged_data[merged_data['STORE_ID'] == int(store)][feature_list].values.reshape(-1, 5)  

    print('train_store_list', train_store_list)       
    # model training
    vae = VAE(args.hid_size, args.hid_layers, args.layer_norm, args.constant_z_scale, args.lr, 
             args.res_struc, [5], args.z_size, np.array([0]*5), sess, 'vae', l2_reg_coeff=args.l2_reg_coeff)
    
    #vae = VAE(args.hid_size, args.hid_layers, [5], args.lr, args.z_size, 'gau', max_category = 6, normal_distribution_index = np.array([0]*5), layer_norm=args.layer_norm, constant_z_scale=args.constant_z_scale, name='vae')

    U.initialize()
    tester.new_saver('vae')
    epochs = args.epochs 
    start_epoch = 0

    best_kld = np.inf


    price_mean = []
    price_std = []
    product_list = merged_data[merged_data['STORE_ID'] == 4245]['UPC'].unique()
    for product in product_list:
        mean = merged_data[merged_data['UPC'] == product]['PRICE'].values.mean()
        std = merged_data[merged_data['UPC'] == product]['PRICE'].values.std()
        price_mean.append(mean)
        price_std.append(std)

    real_action_mean = np.array(price_mean)
    real_action_std = np.array(price_std)

    for epoch in range(start_epoch, epochs):
        logger.info("start learning epoch {}".format(epoch))
        tester.time_step_holder.set_time(epoch)
        if epoch % 100 == 0 and epoch >= 0:
            print('test_data_dict["4245"].shape', test_data_dict["4245"].shape)
            test_code, recons_data = vae.reconstruct_samples(test_data_dict["4245"][0].reshape(-1,5), num_product) #4245
            print('recons_data.shape', recons_data.shape)
            #tester.simple_hist(name='epoch-{}/recons-test-store-{}'.format(epoch, 4245),
            #                   data=[test_data_dict["4245"][0], recons_data], labels=['real', 'fake'],
            #                   density=True, pretty=True,
            #                   xlabel='environment-context', ylabel='density')

            # compute kl
            print('real_action_mean', real_action_mean)
            print('real_action_std', real_action_std)
            print('recons_data', recons_data)
            kld = compute_kl(real_action_mean, real_action_std, recons_data[:, 4])
            print('kld', kld)
            logger.record_tabular("performance/kld", kld)

            if best_kld > kld:
                best_kld > kld
                tester.save_checkpoint(epoch)
                pass

        for _ in range(10):
            train_store = train_store_list[np.random.randint(len(train_store_list))]
            train_week = np.random.randint(len_episode)
            for _ in range(10):
                data_batch = data_dict[train_store]
                data_batch = data_batch[train_week]
                shu_idx = np.arange(data_batch.shape[0])
                np.random.shuffle(shu_idx)
                data_batch = data_batch[shu_idx[:args.sample_product_num]]
                elbo, likelihood, divergence, l2_loss, code = vae.train(data_batch)

        logger.record_tabular("loss/elbo", np.mean(elbo))
        logger.record_tabular("loss/l2_loss", np.mean(l2_loss))
        logger.record_tabular("loss/likelihood", np.mean(likelihood))
        logger.record_tabular("loss/divergence", np.mean(divergence))
        logger.dump_tabular()
