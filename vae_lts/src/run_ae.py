import argparse
import sys

sys.path.append('../../')
from sklearn.decomposition import PCA
import tensorflow as tf
from baselines.common import set_global_seeds
from stable_baselines.common import tf_util as U
from stable_baselines.common.misc_util import boolean_flag
from common import logger
from common.utils import *
from common.config import *
from common.tester import tester
from vae_lts.src.vae_handler import VAE

from lts.src.config import CHOC_BONUS_RANGE


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DEMER")
    # task parameters
    parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--task', help='environment ID', default='lts_ae')
    boolean_flag(parser, 'just_evaluation', default=False)
    # parser.add_argument('--info', help='environment ID', default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    # data load setting
    parser.add_argument('--trans_level', type=int, default=4)
    # vae parameters
    parser.add_argument('--z_size', type=int, default=5)
    boolean_flag(parser, 'share_codes', default=True)
    boolean_flag(parser, 'merge_dim_likelihood', default=True)
    boolean_flag(parser, 'merge_learn', default=True)
    boolean_flag(parser, 'split_z', default=False)
    boolean_flag(parser, 'split_z_by_layer', default=False)
    parser.add_argument('--hid_layers', type=int, default=2)
    parser.add_argument('--l2_reg_coeff', type=float, default=0.1)
    parser.add_argument('--hid_size', type=int, default=512)
    parser.add_argument('--scale_hid', type=int, default=1)
    parser.add_argument('--cond_embedding_layer', type=int, default=0)
    parser.add_argument('--gp_range', type=int, default=8)
    parser.add_argument('--std_level', type=int, default=3)
    parser.add_argument('--sample_user_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00002)
    boolean_flag(parser, 'split_condition_predict', default=False)  # 丢弃参数
    boolean_flag(parser, 'layer_norm', default=True)
    boolean_flag(parser, 'weight_loss', default=False)  # 负面作用
    boolean_flag(parser, 'zero_init', default=True)
    boolean_flag(parser, 'constant_z_scale', default=False)
    boolean_flag(parser, 'cond_with_gau', default=False)
    boolean_flag(parser, 'res_struc', default=False)
    # train parameters
    parser.add_argument('--epochs', type=int, default=10000)

    # boolean_flag(parser, 'full_rank_gau', default=False)
    parser.add_argument('--optimizer', type=str, default=Optimizer.ADAM)
    # test-obj parameters
    parser.add_argument('--load_date', help='if provided, load the model', type=str, default='')
    parser.add_argument('--load_sub_proj', help='if provided, load the model', type=str, default='all-norm-hier_ae')
    parser.add_argument('--max_to_keep_save', help='save model every xx iterations', type=int, default=3)
    parser.add_argument('--log_root', type=str, default='/home/pgao/sim2rec_private/log/')

    args = parser.parse_args()
    kwargs = vars(args)
    tester.set_hyper_param(**kwargs)
    # add track hyperparameter.
    tester.add_record_param(['info',
                             # 'city', # 'evaluate_percent',
                             "z_size",
                             "trans_level",
                             "gp_range",
                             'seed',
                             # "l2_reg_coeff",
                             "std_level",
                             # "sample_user_num",

                             ])
    return args

def main(args):
    from common.tester import tester
    sess = U.make_session(num_cpu=16).__enter__()
    set_global_seeds(args.seed)
    task_name = tester.task_gen([ args.task])
    tester.configure(task_name, 'run_ae', add_record_to_pkl=False, root=args.log_root,
                     max_to_keep=3)


    tester.print_args()
    # preprocess dataset.
    CHOC_BONUS_RANGE[0] = 14 - args.gp_range
    CHOC_BONUS_RANGE[1] = 14 + args.gp_range

    test_choc_mean = (CHOC_BONUS_RANGE[1] - CHOC_BONUS_RANGE[0]) / 2 + CHOC_BONUS_RANGE[0]
    train_choc_mean_list = []
    num_users = 1000
    time_budget = 16
    data_dict = {}
    test_data = np.random.normal(test_choc_mean, args.std_level, size=(time_budget, num_users, 1))
    for i in range(CHOC_BONUS_RANGE[0], CHOC_BONUS_RANGE[1]):
        if np.abs(test_choc_mean - i) > args.trans_level:
            train_choc_mean_list.append(i)
            data_dict[i] = np.random.normal(i, args.std_level, size=(time_budget, num_users, 1))
    vae = VAE(args.hid_size, args.hid_layers, args.layer_norm, args.constant_z_scale, args.lr,
              args.res_struc, [1], args.z_size, np.array([0]), sess, 'vae',
              l2_reg_coeff=args.l2_reg_coeff)
    U.initialize()
    tester.new_saver('vae')
    epochs = args.epochs
    start_epoch = 0
    def compute_kl(real_mean, real_std, recons_data):
        real_logstd = np.log(real_std)
        recons_logstd = np.log(np.std(recons_data))
        recons_std = np.std(recons_data)
        recons_mean = np.mean(recons_data)
        kld = np.sum(recons_logstd - real_logstd + (np.square(real_std) + np.square(real_mean - recons_mean)) / (
                    2 * np.square(recons_std)) - 0.5, axis=-1)
        return kld
    best_kld = np.inf
    for epoch in range(start_epoch, epochs):
        logger.info("start learning epoch {}".format(epoch))
        tester.time_step_holder.set_time(epoch)
        if epoch % 100 == 0 and epoch >= 0:
            # tester.save_checkpoint(epoch)
            test_code, recons_data = vae.reconstruct_samples(test_data[0], num_users)
            tester.simple_hist(name='epoch-{}/recons-test-{}'.format(epoch, test_choc_mean),
                               data=[test_data[0], recons_data], labels=['real', 'fake'],
                               density=True, pretty=True,
                               xlabel='environment-context', ylabel='density')
            # compute kl
            real_std = args.std_level
            real_mean = 14
            kld = compute_kl(real_mean, real_std, recons_data)
            logger.record_tabular("performance/kld", kld)
            if best_kld > kld:
                best_kld = kld
                tester.save_checkpoint(epoch)
                pass

            if epoch % 200 == 0 and epoch >= 0:
                klds = []
                for train_choc_mean in train_choc_mean_list:
                    data_batch = data_dict[train_choc_mean][0]
                    code, recons_data = vae.reconstruct_samples(data_batch, num_users)
                    real_std = args.std_level
                    real_mean = train_choc_mean
                    kld = compute_kl(real_mean, real_std, recons_data)
                    klds.append(kld)
                logger.record_tabular("performance/kld-train", np.mean(klds))
                code_sampling_num = 10
                def get_codes(data_batch):
                    codes = []
                    for i in range(code_sampling_num):
                        shu_idx = np.arange(data_batch.shape[0])
                        np.random.shuffle(shu_idx)
                        data_batch = data_batch[shu_idx[:args.sample_user_num]]
                        code = vae.embedding(data_batch)
                        codes.append(code)
                    return codes
                #from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                test_codes = get_codes(test_data[0])
                all_train_codes = []
                for train_choc_mean in train_choc_mean_list:
                    data_batch = data_dict[train_choc_mean][0]
                    train_codes = get_codes(data_batch)
                    all_train_codes.append(train_codes)

                all_codes = [test_codes] + all_train_codes
                all_codes_flat = np.array(all_codes).reshape((-1, 5))
                # pca.fit(all_codes_flat)
                # all_pca_codes = []
                # for codes in all_codes:
                #     all_pca_codes.append(pca.transform(np.squeeze(codes)))
                trans_codes_flat = pca.fit_transform(all_codes_flat)
                trans_codes = trans_codes_flat.reshape((len(all_train_codes) + 1, code_sampling_num, 2))
                texts = [test_choc_mean] + train_choc_mean_list
                datas = trans_codes
                name = 'code/epoch-{}-pca_code'.format(epoch)
                tester.simple_scatter(name, datas=datas, texts=texts, pretty=True, xlabel='pca dim 1', ylabel='pca dim 2')


            for choc in train_choc_mean_list:
                train_ts = np.random.randint(time_budget)
                train_data_sample = data_dict[choc][train_ts]
                code, recons_data = vae.reconstruct_samples(train_data_sample, num_users)
                tester.simple_hist(name='epoch-{}/recons-train-{}'.format(epoch, choc),
                                   data=[train_data_sample, recons_data], density=True, labels=['real', 'fake'],
                                   pretty=True, xlabel='environment-context', ylabel='density')


            # vae.reconstruct_samples(data, sample_number)
        for _ in range(10):
            train_number = np.random.randint(len(train_choc_mean_list))
            choc_mean = train_choc_mean_list[train_number]
            for _ in range(10):
                train_ts = np.random.randint(time_budget)
                data_batch = data_dict[choc_mean][train_ts]
                shu_idx = np.arange(data_batch.shape[0])
                np.random.shuffle(shu_idx)
                data_batch = data_batch[shu_idx[:args.sample_user_num]]
                elbo, likelihood, divergence, l2_loss, code = vae.train(data_batch)
        logger.record_tabular("loss/elbo", np.mean(elbo))
        logger.record_tabular("loss/l2_loss", np.mean(l2_loss))
        logger.record_tabular("loss/likelihood", np.mean(likelihood))
        logger.record_tabular("loss/divergence", np.mean(divergence))
        logger.dump_tabular()
        # if epoch % 100 == 0 and epoch > 10:
        #


if __name__ == '__main__':
    args = argsparser()
    main(args)
