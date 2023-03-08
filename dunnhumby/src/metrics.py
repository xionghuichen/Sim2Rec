import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true) + 1
    y_pred = np.array(y_pred) + 1
    return np.mean(np.abs((y_true - y_pred)/y_true))

def weighted_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true) + 1e-6
    y_pred = np.array(y_pred) + 1e-6
    return np.sum(np.abs(y_true - y_pred)*y_true) / np.sum(y_true)

def weighted_mean_absolute_error(test, predict):
    test = np.array(test) + 1
    predict = np.array(predict) + 1
    fenmu = max(predict)
    rs = []
    for i in range(len(test)):
        if test[i] == 0:
            p = 1
        else:
            p = test[i]
        fenzi = (abs(test[i] - predict[i]))*p*p
        rs.append(float(fenzi)/fenmu)
    return np.mean(rs)

def get_metrics(y_true, y_pred, param_preflix='validation'):

    wape = weighted_mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2_metric = r2_score(y_true, y_pred)
    mape_metric = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {'wape': np.float(wape), 'rmse': np.float(rmse), 'r2': np.float(r2_metric), 'mape': np.float(mape_metric), 'mae': np.float(mae)}