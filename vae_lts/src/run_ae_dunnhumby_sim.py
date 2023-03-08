import argparse
import xgboost as xgb
import sys

sys.path.append('../../')
import json
import pandas as pd
from sklearn.neighbors import KernelDensity

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from common import logger
from common.utils import *
from common.config import *
# from vae.src.private_config import *
from common.tester import tester
from vae_lts.src.vae_handler import VAE

from lts.src.config import CHOC_BONUS_RANGE
from common.private_config import *
from common.tester import tester
#from sklearn.decomposition import PCA
from lts.src.dunnhumby_env import SingleStoreEnvironment
from lts.src.store_sales_model import StoreSalesModel
import matplotlib.pyplot as plt

def compute_kl(real_mean, real_std, recons_data):
    real_logstd = np.log(real_std)
    recons_logstd = np.log(np.std(recons_data))
    recons_std = np.std(recons_data)
    recons_mean = np.mean(recons_data)
    kld = np.sum(recons_logstd - real_logstd + (np.square(real_std) + np.square(real_mean - recons_mean)) / (
                2 * np.square(recons_std)) - 0.5, axis=-1)
    return kld


def compute_kl_real(real_data, fake_data, bandwidth, epoch):
    if fake_data.shape[0] == 0:
        return None, None, None
    shuffle_index = np.random.randint(0, real_data.shape[0], min(2000, real_data.shape[0]))
    fake_shuffle_index = np.random.randint(0, fake_data.shape[0], min(2000, fake_data.shape[0]))
    print('real_data[shuffle_index]', real_data[shuffle_index])
    print('real_data[shuffle_index].shape', real_data[shuffle_index].shape)
    print('real_data.shape', real_data.shape)
    kde_real = KernelDensity(bandwidth=bandwidth, rtol=5e-2).fit(real_data[shuffle_index])
    kde_fake = KernelDensity(bandwidth=bandwidth, rtol=5e-2).fit(fake_data[fake_shuffle_index])
    score_real = kde_real.score_samples(real_data[shuffle_index])
    score_fake = kde_fake.score_samples(real_data[shuffle_index])
    fig, ax = plt.subplots()
    ax.plot(real_data[shuffle_index], np.exp(score_real), label='real data')
    ax.plot(real_data[shuffle_index], np.exp(score_fake), label='fake data')
    ax.legend()
    ax.set_title('histogram at epoch {}'.format(epoch))
    fig.savefig('/home/ynmao/sim_rec_tf1/sim2rec_private_new/sim2rec/vae_lts/src/histogram_figures/histogram at epoch {}.png'.format(epoch))
    kl = np.mean(np.clip(score_real - score_fake, None, 20))
    return kl

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DEMER")
    #task parameters
    parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--task', help='environment ID', default='lts_ae')
    boolean_flag(parser, 'just_evaluation', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)

    #data load setting
    parser.add_argument('--train_state_list', type=str, default=['KY']) #['KY', 'OH', 'TX']
    parser.add_argument('--test_state_list', type=str, default=['IN']) #['IN', 'KY']
    parser.add_argument('--direct_trans_state_list', type=str, default=['OH'])

    #vae parameters
    parser.add_argument('--z_size', type=int, default=5)
    parser.add_argument('--hid_layers', type=int, default=2)
    parser.add_argument('--l2_reg_coeff', type=float, default=0.01)
    parser.add_argument('--hid_size', type=int, default=512)
    parser.add_argument('--sample_product_num', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00002)
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
    sampling_trajectory = []
    for i in range(10):
        trajectory = env.simulate_trajectory()
        sampling_trajectory.append(trajectory)
    sampling_trajectory = np.concatenate(np.array(sampling_trajectory), axis=0)
    return sampling_trajectory




if __name__ == '__main__':
    args = argsparser()

    merged_data = pd.read_csv('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/merged_data_norm_1022.csv')
    with open('/home/ynmao/sim_rec_tf1/dunnhumby/data/processed/store_state.json') as f:
        store_state = json.load(f)

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
    dim_feature = 12 + 1
    env_dict = {}
    data_dict = {}
    test_data_dict = {}


    # preprocess dataset
    
    # train
    for store, state in store_state.items():
        if state in args.train_state_list:
            print('store', store)
            train_store_list.append(store)
            store_model = StoreSalesModel(store, merged_data, store_state)
            store_env = SingleStoreEnvironment(store, store_model)
            env_dict[store] = store_env
            data_dict[store] = vae_data_info_preprocess(store_env)[:,:,3:]
            print('data_dict[store]', data_dict[store])

    for store, state in store_state.items():
        if state in args.test_state_list:
            print('store', store)
            store_model = StoreSalesModel(store, merged_data, store_state)
            store_env = SingleStoreEnvironment(store, store_model)
            env_dict[store] = store_env
            test_data_dict[store] = vae_data_info_preprocess(store_env)[:,:,3:]
            print('test_data_dict[store]', test_data_dict[store])
    
    print('train_store_list', train_store_list)
           
    # model training
    vae = VAE(args.hid_size, args.hid_layers, args.layer_norm, args.constant_z_scale, args.lr, 
              args.res_struc, [6], args.z_size, np.array([0]*6), sess, 'vae', l2_reg_coeff=args.l2_reg_coeff)

    U.initialize()
    tester.new_saver('vae')
    epochs = args.epochs 
    start_epoch = 0

    best_kld = np.inf

    for epoch in range(start_epoch, epochs):
        logger.info("start learning epoch {}".format(epoch))
        tester.time_step_holder.set_time(epoch)
        if epoch % 100 == 0 and epoch >= 0:
            test_code, recons_data = vae.reconstruct_samples(test_data_dict["4245"][0], num_product) #4245
            print('recons_data.shape', recons_data.shape)
            #tester.simple_hist(name='epoch-{}/recons-test-store-{}'.format(epoch, 4245),
            #                   data=[test_data_dict["4245"][0], recons_data], labels=['real', 'fake'],
            #                   density=True, pretty=True,
            #                   xlabel='environment-context', ylabel='density')

            # compute kl
            real_action_mean = env_dict["4245"].price_mean
            real_action_std = env_dict["4245"].price_std
            real_action = env_dict["4245"].price
            print('real_action_mean', real_action_mean)
            print('real_action_std', real_action_std)
            print('recons_data[:, 3]', recons_data[:, 3]) #3: price 5:sales
            kld = compute_kl(real_action_mean, real_action_std, recons_data[:, 3])
            #kld = compute_kl_real(real_action, recons_data[:, 3], 1.0, epoch)
            print('kld', kld)
            logger.record_tabular("performance/kld", kld)

            if best_kld > kld:
                best_kld = kld
                tester.save_checkpoint(epoch)
                pass

        elbo_list = []
        likelihood_list = []
        divergence_list = []
        l2_loss_list = []
        #for _ in range(400):
        for train_store in train_store_list:
            for train_week in range(len_episode*10):
                #train_store = train_store_list[np.random.randint(len(train_store_list))]
                #train_week = np.random.randint(len_episode*10)
                #print('train_store', train_store)
                #print('train_week', train_week)
                #for _ in range(10):
                data_batch = data_dict[train_store]
                data_batch = data_batch[train_week]
                data_batch = data_batch[:55,:]
                    #shu_idx = np.arange(data_batch.shape[0])
                    #np.random.shuffle(shu_idx)
                    #print('shu_idx', shu_idx)
                    #data_batch = data_batch[shu_idx[:args.sample_product_num]]
                    #data_batch = data_batch[shu_idx[:args.sample_product_num]]
                elbo, likelihood, divergence, l2_loss, code = vae.train(data_batch)
                elbo_list.append(elbo)
                likelihood_list.append(likelihood)
                divergence_list.append(divergence)
                l2_loss_list.append(l2_loss)

        np_elbo_list = np.array(elbo_list)
        np_likelihood_list = np.array(likelihood_list)
        np_divergence_list = np.array(divergence_list)
        np_l2_loss_list = np.array(l2_loss_list)

        #logger.record_tabular("loss/elbo", np.mean(elbo))
        #logger.record_tabular("loss/l2_loss", np.mean(l2_loss))
        #logger.record_tabular("loss/likelihood", np.mean(likelihood))
        #logger.record_tabular("loss/divergence", np.mean(divergence))
        logger.record_tabular("loss/elbo", np.mean(np_elbo_list))
        logger.record_tabular("loss/l2_loss", np.mean(np_l2_loss_list))
        logger.record_tabular("loss/likelihood", np.mean(np_likelihood_list))
        logger.record_tabular("loss/divergence", np.mean(np_divergence_list))
        logger.dump_tabular()
