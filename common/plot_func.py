# from baselines.results_plotter import *
import matplotlib.pyplot as plt
import numpy as np
from common import plot_util


def split_by_task(taskpath, split_keys, y_names):
    pair_delimiter = '&'
    kv_delimiter = '='
    pairs = taskpath.dirname.split(pair_delimiter)
    # value = []
    key_value = {}
    for p in pairs:
        key = kv_delimiter.join(p.split(kv_delimiter)[:-1])
        key_value[key] = p.split(kv_delimiter)[-1]
    # filter_key_value = {}
    parse_list = []
    for split_key in split_keys:
        if split_key in key_value.keys():
            parse_list.append(split_key + '=' + key_value[split_key])
            # filter_key_value[split_key] = key_value[split_key]
        else:
            parse_list.append(split_key + '=NF')
    task_split_key = '.'.join(parse_list)
    split_keys = []
    for y_name in y_names:
        split_keys.append(task_split_key + ' eval:' + y_name)
    return split_keys, y_names

    # if y_names is not None:
    #     split_keys = []
    #     for y_name in y_names:
    #         split_keys.append(task_split_key+' eval:' + y_name)
    #     return split_keys, y_names
    # else:
    #     return task_split_key, y_names
    # return '_'.join(value[-3:])

def auto_gen_key_value_name(dict):
    parse_list = []
    for key, value in dict.iterms():
        parse_list.append(key + '=' + value)


def picture_split(taskpath, single_name=None, split_keys=None, y_names=None):
    if single_name is not None:
        return single_name, None
    else:
        return split_by_task(taskpath, split_keys, y_names)

def csv_to_xy(r, x_name, y_name, scale_dict, x_bound=None, x_start=None, y_bound=None, remove_outlier=False):

    df = r.progress.copy().reset_index() # ['progress']
    if y_name not in list(df.columns):
        return None
    df.drop(df[np.isnan(df[x_name])].index, inplace=True)
    df.drop(df[np.isnan(df[y_name])].index, inplace=True)
    # pd = pd.dropna(axis=0, how='any')
    x = df[x_name]
    y = df[y_name]
    if x_bound is None:
        x_bound = x.max()
    if x_start is None:
        x_start = x.min()
    filter_index = (x <= x_bound) & (x >= x_start)
    x = x[filter_index]
    y = y[filter_index]
    if y_bound is not None:
        y[y > y_bound] = y_bound
    if remove_outlier:
        z_score = (y - y.mean()) / y.std()
        filter_index = z_score < 10.0
        x = x[filter_index]
        y = y[filter_index]

    y = y * scale_dict[y_name]
    return x, y

if __name__=='__main__':
    # dirs = '../log/vX/Weifang'
    # # dirs = '../log/vX/Weifang/trpo[multi head]Driver-vX-DEMER.batch_size_4096.g_step_4.hidden_layers_3.seed_0.date(20190506, 20190526).confounder_False.ha_2t.2019-06-05.23-22-55'
    # results = plot_util.load_results(dirs,  enable_monitor=False)
    # y_names = ['loss/d/expert_acc', 'optimgain']
    # plot_util.plot_results(results, xy_fn= lambda r, y_name: csv_to_xy(r, 'info/TimestepsSoFar', y_name),
    #                        # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
    #                        split_fn=lambda r: picture_split(taskpath=r, single_name='test')[0],
    #                        group_fn=lambda r: picture_split(taskpath=r, split_keys=['ha', 'confounder'], y_names=y_names),
    #                        average_group=True, resample=int(1e3))
    dirs = '../log/vX/Weifang_online/'
    results = plot_util.load_results(dirs, enable_monitor=False)
    y_names = ['loss/d/expert_acc', 'optimgain']
    # eval_policy/coupon_avg_rate
    postfixs = ['coupon_avg_rate', 'coupon_predict_rate', 'roi_avg', 'roi_predict', 'sum_avg_gmv', 'sum_fos',
                'sum_predict_gmv', 'sum_spend']
    for postfix in postfixs:
        y_names = ['eval_policy/' + postfix, 'eval_real/' + postfix, 'eval_zero/' + postfix]
        plot_util.plot_results(results, xy_fn= lambda r, y_name: csv_to_xy(r, 'info/TimestepsSoFar', y_name),
                               # xy_fn=lambda r: ts2xy(r['monitor'], 'info/TimestepsSoFar', 'diff/driver_1_2_std'),
                               split_fn=lambda r: postfix + picture_split(taskpath=r, split_keys=['eval_stochastic'], y_names=y_names)[0],
                               group_fn=lambda r: picture_split(taskpath=r, y_names=y_names),
                               shaded_std=False,
                               average_group=True, resample=int(1e3))
        plt.show()
