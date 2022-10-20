from common import logger
from common.utils import *


class Statistic(object):
    PREDICT_GMV = 'predict_gmv'
    AVG_GMV = 'avg_gmv'
    FOS = 'fos'
    SPEND = 'spend'
    COUPON_INFO = 'coupon_info'

    def __init__(self, name, revise_func=None, save_pic_location=None, iters=None):
        self.name = name
        self.coupon_avg_rate = None
        self.driver_fos_mean = None
        self.driver_fos_std = None
        self.save_pic_location = save_pic_location
        self.iters = iters
        self.revise_func = revise_func
        self.init()

    def new_episode(self):
        self.epi += 1
        self.total_predict_gmv.append([])
        self.total_avg_gmv.append([])
        self.total_fos.append([])
        self.total_spend.append([])
        self.total_coupon_info.append([])
        self.total_revise_coupon_info.append([])

    def pop(self):
        self.total_predict_gmv.pop()
        self.total_avg_gmv.pop()
        self.total_fos.pop()
        self.total_spend.pop()
        self.total_coupon_info.pop()
        self.total_revise_coupon_info.append([])

    def reshape_episode(self):
        if len(self.total_predict_gmv) == 1:
            self.total_predict_gmv = self.total_predict_gmv[0]
            self.total_avg_gmv = self.total_avg_gmv[0]
            self.total_fos = self.total_fos[0]
            self.total_spend = self.total_spend[0]
            self.total_coupon_info = self.total_coupon_info[0]
            if self.revise_func is not None:
                self.total_revise_coupon_info = self.total_revise_coupon_info[0]
        else:
            self.total_predict_gmv = np.array(self.total_predict_gmv).T
            self.total_avg_gmv = np.array(self.total_avg_gmv).T
            self.total_fos = np.array(self.total_fos).T
            self.total_spend = np.array(self.total_spend).T
            self.total_coupon_info = np.array(self.total_coupon_info).transpose(1, 0, 2)
            if self.revise_func is not None:
                self.total_revise_coupon_info = np.array(self.total_revise_coupon_info).transpose(1, 0, 2)

    def append(self, info):
        assert isinstance(info, dict)
        info = info[self.name]
        self.total_avg_gmv[self.epi].append(info[self.AVG_GMV])
        self.total_fos[self.epi].append(info[self.FOS])
        # self.total_predict_gmv[self.epi].append(info[self.PREDICT_GMV])
        self.total_spend[self.epi].append(info[self.SPEND])
        self.total_coupon_info[self.epi].append(info[self.COUPON_INFO])
        if self.revise_func is not None:
            reshape_coupon_info = self.revise_func(info[self.COUPON_INFO])
            self.total_revise_coupon_info[self.epi].append(reshape_coupon_info)

    def init(self):
        self.total_predict_gmv = [[]]
        self.total_avg_gmv = [[]]
        self.total_fos = [[]]
        self.total_spend = [[]]
        self.total_coupon_info = [[]]
        self.total_revise_coupon_info = [[]]
        self.epi = 0

    def compute_statistic(self, prefix):
        sum_avg_gmv = np.sum(self.total_avg_gmv)
        # sum_predict_gmv = np.sum(self.total_predict_gmv)
        sum_fos = np.sum(self.total_fos)
        sum_spend = np.sum(self.total_spend)
        coupon_avg_rate = sum_spend / sum_avg_gmv
        # coupon_predict_rate = sum_spend / sum_predict_gmv
        coupon_percentage = self.coupon_percentage()  # np.unique(np.where(total_spend > 0)[0]).shape[0] / total_spend.shape[0]
        dim_coupon_info_mean = self.dim_coupon_info_mean()
        prefix = prefix + '_' + self.name + '/'
        for i in range(dim_coupon_info_mean.shape[0]):
            logger.record_tabular(prefix + "{}_dim_coupon_info_mean".format(i), dim_coupon_info_mean[i])
        logger.record_tabular(prefix + "coupon_percentage", coupon_percentage)
        logger.record_tabular(prefix + "avg_gmv", sum_avg_gmv)
        # logger.record_tabular(prefix + "predict_gmv", sum_predict_gmv)
        logger.record_tabular(prefix + "fos", sum_fos)
        logger.record_tabular(prefix + "spend", sum_spend)
        logger.record_tabular(prefix + "coupon_avg_rate", coupon_avg_rate)
        # logger.record_tabular(prefix + "coupon_predict_rate", coupon_predict_rate)
        self.coupon_avg_rate = coupon_avg_rate
        self.driver_fos_mean = np.mean(self.total_fos, axis=0)
        self.driver_fos_std = np.std(self.total_fos, axis=0)

    def coupon_percentage(self):

        total_spend = np.array(self.total_spend)
        coupon_info = np.array(self.total_coupon_info)
        coupon_percentage = np.unique(np.where(np.all(coupon_info[:, :, 0:2] > 0.01, axis=2))[1]).shape[0] / float(
            total_spend.shape[1])
        return coupon_percentage

    def dim_coupon_info_mean(self):
        """
        compute the average value for each dimension of (raw) coupon information where the spend not equal to zero.
        :return:
        """
        total_spend = np.array(self.total_spend)
        # coupon_info = np.array(self.total_coupon_info)[np.where(total_spend > 0.01)]
        coupon_info = np.array(self.total_coupon_info)
        coupon_info = coupon_info[np.where(np.all(coupon_info[:, :, 0:2] > 0.01, axis=2))]
        dim_coupon_info_mean = np.mean(coupon_info, axis=0)
        return dim_coupon_info_mean

    def compute_ab_test_unit_info(self, axis):
        sum_total_avg_gmv = np.expand_dims(np.sum(np.asarray(self.total_avg_gmv), axis=axis), -1)
        sum_fos = np.expand_dims(np.sum(np.asarray(self.total_fos), axis=axis), -1)
        sum_spend = np.expand_dims(np.sum(np.asarray(self.total_spend), axis=axis), -1)
        return sum_total_avg_gmv, sum_fos, sum_spend

    def compute_ab_test_stat_unit(self, target_statistic, ano_group_statistic, ano_group_target_statistic, prefix, axis, axis_name):

        avg_gmv, fos, spend = self.compute_ab_test_unit_info(axis)
        target_avg_gmv, target_fos, target_spend = target_statistic.compute_ab_test_unit_info(axis)
        ano_group_avg_gmv, ano_group_fos, ano_group_spend = ano_group_statistic.compute_ab_test_unit_info(axis)
        ano_group_target_avg_gmv, ano_group_target_fos, ano_group_target_spend = ano_group_target_statistic.compute_ab_test_unit_info(axis)

        roi = np.clip((target_avg_gmv - ano_group_target_avg_gmv) / ((target_spend) + 1), -5, 5)
        predict_roi = np.clip((avg_gmv - ano_group_avg_gmv) / ((spend) + 1), -5, 5)

        fos_inc = (target_fos - ano_group_target_fos)
        predict_fos_inc = (fos - ano_group_fos)
        spend_inc = (target_spend - ano_group_target_spend)
        predict_spend_inc = (spend - ano_group_spend)
        prefix = prefix + '_' + self.name + '/'
        logger.record_tabular(prefix + "{}.ROI.data_amount".format(axis_name), avg_gmv.shape[0])
        logger.record_tabular(prefix + "{}.ROI.real".format(axis_name), np.mean(roi))
        logger.record_tabular(prefix + "{}.ROI.predict".format(axis_name), np.mean(predict_roi))
        logger.record_tabular(prefix + "{}.ROI.MAE".format(axis_name), mae(roi, predict_roi))
        logger.record_tabular(prefix + "{}.ROI.MAPE".format(axis_name), mape(roi, predict_roi))

        logger.record_tabular(prefix + "{}.FOS_INC.real".format(axis_name), np.mean(fos_inc))
        logger.record_tabular(prefix + "{}.FOS_INC.predict".format(axis_name), np.mean(predict_fos_inc))
        logger.record_tabular(prefix + "{}.FOS_INC.MAE".format(axis_name), mae(fos_inc, predict_fos_inc))
        logger.record_tabular(prefix + "{}.FOS_INC.MAPE".format(axis_name), mape(fos_inc, predict_fos_inc))
        # logger.record_tabular(prefix + "{}.FOS_INC-MAE-MAPE".format(axis_name), mae_mape(fos_inc, predict_fos_inc, 10))

        logger.record_tabular(prefix + "{}.SPEND_INC.real".format(axis_name), np.mean(spend_inc))
        logger.record_tabular(prefix + "{}.SPEND_INC.predict".format(axis_name), np.mean(predict_spend_inc))
        logger.record_tabular(prefix + "{}.SPEND_INC.MAE".format(axis_name), mae(spend_inc, predict_spend_inc))
        logger.record_tabular(prefix + "{}.SPEND_INC.MAPE".format(axis_name), mape(spend_inc, predict_spend_inc))
        # logger.record_tabular(prefix + "{}.SPEND_INC-MAE-MAPE".format(axis_name), mae_mape(spend_inc, predict_spend_inc, 10))


    def compute_abtest_statistic(self, target_statistic, ano_group_statistic, ano_group_target_statistic, prefix):
        assert isinstance(target_statistic, Statistic)
        assert isinstance(ano_group_statistic, Statistic)
        assert isinstance(ano_group_target_statistic, Statistic)
        # self.compute_ab_test_stat_unit(target_statistic, ano_group_statistic, ano_group_target_statistic, prefix, 0, 'dri')
        self.compute_ab_test_stat_unit(target_statistic, ano_group_statistic, ano_group_target_statistic, prefix, 1, 'day')
        self.compute_ab_test_stat_unit(target_statistic, ano_group_statistic, ano_group_target_statistic, prefix, (0, 1), 'tot')


    def compare_statistic_unit(self, target_statistic, prefix, axis, axis_name, quality_method):
        target_sum_total_avg_gmv = np.expand_dims(np.sum(np.asarray(target_statistic.total_avg_gmv), axis=axis), -1)
        sum_total_avg_gmv = np.expand_dims(np.sum(np.asarray(self.total_avg_gmv), axis=axis), -1)
        target_sum_spend = np.expand_dims(np.sum(np.asarray(target_statistic.total_spend), axis=axis), -1)
        sum_spend = np.expand_dims(np.sum(np.asarray(self.total_spend), axis=axis), -1)
        target_sum_coupon_info = np.expand_dims(np.mean(np.asarray(target_statistic.total_coupon_info), axis=axis), -1)
        sum_coupon_info = np.expand_dims(np.mean(np.asarray(self.total_coupon_info), axis=axis), -1)

        prefix = prefix + '_' + self.name + '/'
        max_fos = np.max(target_statistic.total_fos)
        driver_avg_fos = np.expand_dims(np.sum(np.asarray(target_statistic.total_fos), axis=(0)), -1)
        index_list = []
        last_fos = 0

        for cur_fos in range(10, max_fos + 10, 10):
            index_list.append({
                "last_fos": last_fos,
                "cur_fos": cur_fos,
                "idx": np.where(np.logical_and(driver_avg_fos[:, 0] >= last_fos , driver_avg_fos[:, 0] < cur_fos))[0]
            })

            last_fos += 10
        index_list.append({
            "last_fos": 0,
            "cur_fos": 100,
            "idx": np.where(np.logical_and(driver_avg_fos[:, 0] >= last_fos , driver_avg_fos[:, 0] < max_fos + 10))[0]
        })

        for index_dict in index_list:
            last_fos = index_dict['last_fos']
            cur_fos = index_dict['cur_fos']
            idx = index_dict['idx']
            target_sum_fos = np.expand_dims(np.sum(np.asarray(target_statistic.total_fos)[:, idx], axis=axis), -1)
            sum_fos = np.expand_dims(np.sum(np.asarray(self.total_fos)[:, idx], axis=axis), -1)
            target_mean_fos = np.expand_dims(np.mean(np.asarray(target_statistic.total_fos)[:, idx], axis=axis), -1)
            mean_fos = np.expand_dims(np.mean(np.asarray(self.total_fos)[:, idx], axis=axis), -1)
            logger.record_tabular(prefix + "MAE.{}-{}.{}.fos".format(last_fos, cur_fos, axis_name), mae(target_mean_fos, mean_fos))
            logger.record_tabular(prefix + "MAE.{}-{}.{}.fos_sum".format(last_fos, cur_fos, axis_name), mae(target_sum_fos, sum_fos))
            logger.record_tabular(prefix + "MAPE.{}-{}.{}.fos".format(last_fos, cur_fos, axis_name), mape(target_sum_fos, sum_fos))
            logger.record_tabular(prefix + "MAPE.{}-{}.{}.spend".format(last_fos, cur_fos, axis_name), mape(target_sum_spend, sum_spend))
            logger.record_tabular(prefix + "MAE.{}-{}.{}.amount".format(last_fos, cur_fos, axis_name),
                                  idx.shape[0])
        if quality_method == 'ar^2':
            logger.record_tabular(prefix + "AR^2.{}.coupon_info".format(axis_name),
                                  compute_adjust_r2(target_sum_coupon_info, sum_coupon_info))
        elif quality_method == 'diff':
            coupon_percentage = self.coupon_percentage()
            target_coupon_percentage = target_statistic.coupon_percentage()
            target_dim_coupon_info_mean = target_statistic.dim_coupon_info_mean()
            dim_coupon_info_mean = self.dim_coupon_info_mean()
            logger.record_tabular(prefix + "diff.{}.cp_percentage".format(axis_name),
                                  np.squeeze(coupon_percentage - target_coupon_percentage))
            for i in range(target_dim_coupon_info_mean.shape[0]):
                logger.record_tabular(prefix + "diff.{}.{}.coupon_info".format(axis_name, i),
                                      np.squeeze(sum_coupon_info[i] - target_sum_coupon_info[i]))
                logger.record_tabular(prefix + "diff.{}.!0.{}._coupon_info".format(axis_name, i),
                                      np.squeeze(dim_coupon_info_mean[i] - target_dim_coupon_info_mean[i]))

    def compare_statistic(self, target_statistic, prefix):
        assert isinstance(target_statistic, Statistic)
        self.compare_statistic_unit(target_statistic, prefix, 1, 'day', 'ar^2')
        self.compare_statistic_unit(target_statistic, prefix, 0, 'dri', 'ar^2')
        self.compare_statistic_unit(target_statistic, prefix, (0, 1), 'tot', 'diff')
        target_sum_fos = np.asarray(target_statistic.total_fos).reshape([-1, 1])
        sum_fos = np.asarray(self.total_fos).reshape([-1, 1])
        prefix = prefix + '_' + self.name + '/'
        logger.record_tabular(prefix + "MAE&MAPE.fos", mae_mape(target_sum_fos, sum_fos, 10))
        logger.record_tabular(prefix + "MAE.{single}.fos", mae(target_sum_fos, sum_fos))
        logger.record_tabular(prefix + "MAPE.{single}.fos", mape(target_sum_fos, sum_fos))
        source_fos_num = []
        target_fos_num = []
        np_target_fos = np.asarray(target_statistic.total_fos).astype('int32')
        np_fos = np.asarray(self.total_fos).astype('int32')
        assert np_target_fos.shape[0] + np_target_fos.shape[1] == np_fos.shape[0] + np_fos.shape[1]
        total_drivers = np_target_fos.shape[0] * np_target_fos.shape[1]
        max_fo = np.max(np_target_fos)
        min_fo = 0
        index_list = []
        for i in range(min_fo, max_fo):
            index_list.append(i)
            target_fos_num.append(np.where(np_target_fos==i)[0].shape[0])
            source_fos_num.append(np.where(np_fos==i)[0].shape[0])
        source_fos_num = np.array(source_fos_num)
        target_fos_num = np.array(target_fos_num)
        diff_fos_num_percent = np.mean(np.clip(np.abs(source_fos_num - target_fos_num)/(target_fos_num + 0.1), -5, 5))
        diff_fos_num_dist = np.mean(np.abs(source_fos_num/total_drivers - target_fos_num/total_drivers))

        logger.record_tabular(prefix + "fos_num_percent.tot", diff_fos_num_percent)
        logger.record_tabular(prefix + "fos_num_dist.tot", diff_fos_num_dist)

        print("save_pic_location {}".format(self.save_pic_location))
        if self.save_pic_location is not None:
            import os
            os.makedirs(self.save_pic_location, exist_ok=True)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.bar(index_list, np.clip((source_fos_num - target_fos_num)/(target_fos_num + 0.1), -3, 3), yerr=0)
            # plt.title(self.save_pic_location.split('/')[-1])
            plt.title("diff_percent_err-{}.png".format(self.iters))
            plt.ylim(-1, 3)
            plt.savefig(self.save_pic_location + 'diff_percent_err-{}.png'.format(self.iters), dpi=100)
            plt.figure()
            plt.bar(index_list, (source_fos_num / total_drivers - target_fos_num /total_drivers), yerr=0)
            plt.title("diff_dist_err-{}".format(self.iters))
            plt.ylim(-0.2, 0.2)
            plt.savefig(self.save_pic_location + 'diff_dist_err-{}.png'.format(self.iters), dpi=100)
        prefore_index = 0
        for i in range(10, max_fo - min_fo, 10):
            source_fos_num_part = source_fos_num[prefore_index:i]
            target_fos_num_part = target_fos_num[prefore_index:i]
            diff_fos_num_percent_part = np.clip(np.mean(np.abs(source_fos_num_part - target_fos_num_part)/(target_fos_num_part + 0.1)), -5, 5)
            diff_fos_num_dist_part = np.mean(np.abs(source_fos_num_part / total_drivers - target_fos_num_part / total_drivers))

            logger.record_tabular(prefix + "fos_num_percent.{}-{}".format(prefore_index + min_fo, i + min_fo), diff_fos_num_percent_part)
            logger.record_tabular(prefix + "fos_num_dist.{}-{}".format(prefore_index + min_fo, i + min_fo), diff_fos_num_dist_part)
            prefore_index = i


    def find_high_frequency_coupon_info(self, amount, min_percent=0.95, rescale_ratio=2):
        if self.revise_func is not None:
            logger.info("use revise func")
            coupon_info = np.array(self.total_revise_coupon_info)
        else:
            coupon_info = np.array(self.total_coupon_info)
        return find_high_frequency_coupon_info(coupon_info, amount, min_percent, rescale_ratio)


def find_high_frequency_coupon_info(coupon_info, amount, min_percent=0.95, rescale_ratio=2):
    reshape_coupon_info = coupon_info.reshape([-1, 5])
    # 精确到角
    reshape_coupon_info[:, :4] = (reshape_coupon_info[:, :4] * rescale_ratio).astype(
        'int32') / rescale_ratio  # np.round(reshape_coupon_info, 1)
    # reshape_coupon_info[:, 4] = np.round(reshape_coupon_info[:, 4], 4)
    delete_zero_cp = reshape_coupon_info[np.where(np.all(reshape_coupon_info[:, 0:2] > 0.01, axis=1))]

    if delete_zero_cp.shape[0] == 0:
        logger.info("empty for not-zero coupon info")
        res_coupon_info = np.array([[0, 0., 0, 0., 0.]])
    else:
        delete_zero_uni_cp = np.unique(delete_zero_cp, axis=0)
        logger.info("uniq cp :{}".format(delete_zero_uni_cp.shape))
        # delete_zero_cp[np.where(np.all(delete_zero_cp == delete_zero_uni_cp[0], axis=1))]
        counts = []
        coupon_infos = []
        for unicp in delete_zero_uni_cp:
            counts.append(delete_zero_cp[np.where(np.all(delete_zero_cp == unicp, axis=1))].shape[0])
            coupon_infos.append(unicp)
        freq = np.array(counts) / np.sum(counts)
        coupon_infos = np.array(coupon_infos)
        sort_freq = np.sort(freq)[::-1]
        sort_freq_index = np.argsort(freq)[::-1]
        high_freq_coupon_info = coupon_infos[sort_freq_index]
        high_freq_number = sort_freq
        total_freq = 0
        count = 0
        res_coupon_info = []
        onetime = False
        onetime2 = False
        onetime3 = False
        for hfci, sf in zip(high_freq_coupon_info, high_freq_number):
            logger.info("high_freq_coupon_info {}, freq {}".format(hfci, sf))
            total_freq += sf
            res_coupon_info.append(hfci)
            count += 1
            if total_freq > 0.99 and not onetime:
                logger.record_tabular("coupon_type/" + self.name + '-0.99', count)
                onetime = True
            elif total_freq > 0.95 and not onetime2:
                onetime2 = True
                logger.record_tabular("coupon_type/" + self.name + '-0.95', count)
            elif total_freq > 0.9 and not onetime3:
                onetime3 = True
                logger.record_tabular("coupon_type/" + self.name + '-0.9', count)
            if total_freq > min_percent and count >= amount:
                break
            if count > 3000:
                break
        logger.info("cover freq {}. amount {}".format(total_freq, count))
        res_coupon_info = np.append(res_coupon_info, [[0, 0., 0, 0., 0.]], axis=0)
    return res_coupon_info
