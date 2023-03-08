from datetime import datetime
import numpy as np

def compute_days(start_date, end_date, format):
    start_date = datetime.strptime(start_date, format)
    end_date = datetime.strptime(end_date, format)
    return (end_date - start_date).days

def compute_adjust_r2(real_data, predict_data):
    real_data = np.asarray(real_data)
    predict_data = np.asarray(predict_data)
    reduce_axis = list(range(len(real_data.shape) - 1))
    data_shape = np.array(real_data.shape, np.float)
    data_number = np.sum(data_shape[:-1])
    feature_number = data_shape[-1]
    r2_error = np.sum(np.abs(real_data - predict_data)) / np.sum(np.abs(real_data - np.mean(real_data, axis=tuple(reduce_axis))))
    ad_r2 = 1 - r2_error * (data_number - 1) / (data_number - feature_number - 1)
    return ad_r2

def mae(real_data, predict_data):
    real_data = np.asarray(real_data)
    predict_data = np.asarray(predict_data)
    assert real_data.shape == predict_data.shape
    if len(real_data.shape) > 1:
        assert real_data.shape[1] == 1
        return np.mean(np.abs(predict_data - real_data))
    else:
        return np.mean(np.abs(predict_data - real_data))

def mape(real_data, predict_data):
    real_data = np.asarray(real_data)
    predict_data = np.asarray(predict_data)
    assert real_data.shape == predict_data.shape
    if len(real_data.shape) > 1:
        assert real_data.shape[1] == 1
        return np.mean(np.clip(np.abs(predict_data - real_data)/np.abs((real_data + 0.0001)), -2, 2))
    else:
        return np.mean(np.clip(np.abs(predict_data - real_data)/np.abs((real_data + 0.0001)), -2, 2))

def mae_mape(real_data, predict_data, threshold):
    real_data = np.asarray(real_data)
    predict_data = np.asarray(predict_data)
    assert real_data.shape == predict_data.shape
    assert real_data.shape[1] == 1
    mae_idx = np.where(real_data < threshold)
    ae = np.abs(predict_data[mae_idx] - real_data[mae_idx])
    mape_idx = np.where(real_data >= threshold)
    ape = np.clip(np.abs(predict_data[mape_idx] - real_data[mape_idx])/np.abs((real_data[mape_idx] + 0.0001)), -2, 2)
    return np.concatenate([ae, ape]).mean()


def prefix_search(search_list, prefix):
    return [i for i in range(len(search_list)) if search_list[i].startswith(prefix)]

def auto_parameters_str_gen(keys, **kwargs):
    hyper_param_record = []
    for k in keys:
        hyper_param_record.append(str(k) + '=' + str(kwargs[k]))
    return '&'.join(hyper_param_record)

def get_mean_std(rescale, mean, std):
    if rescale:
        return mean, std
    else:
        mean = np.zeros(mean.shape)
        std = np.ones(std.shape)
    return mean, std


def get_index_from_names(target_list, filter_list):
    target_list = target_list.tolist()
    indeics = []
    for fi in filter_list:
        try:
            indeics.append(target_list.index(fi))
        except Exception as e:
            pass
    indeics = sorted(indeics)
    # indeics = np.sort(indeics)
    return indeics

def box_concate(box1, box2):
    import gym
    assert isinstance(box1, gym.spaces.Box) and isinstance(box2, gym.spaces.Box)
    assert box1.dtype == box2.dtype
    #print('box1.low', box1.low)
    #print('box1.high', box1.high)
    #print('box2.low', box2.low)
    #print('box2.high', box2.high)
    low = np.concatenate([box1.low, box2.low], axis=1)
    high = np.concatenate([box1.high, box2.high], axis=1)
    return gym.spaces.Box(low=low, high=high, dtype=box1.dtype, )

if __name__ == '__main__':
    target_list = ['1', '2', '3', 'd', '5', 'a', '10']
    filter_list = ['d', 'a']
    print(get_index_from_names(target_list, filter_list))