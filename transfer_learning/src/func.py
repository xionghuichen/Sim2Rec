import numpy as np


def auto_train_test_split(city_list, relation_matrix, test_city_size):
    print("relation matrix shape {}".format(relation_matrix.shape))
    average_relation = np.mean(relation_matrix, axis=(1, 2, 3))
    sort_index = np.argsort(average_relation)
    test_city_list = list(np.array(city_list)[sort_index[-1 * test_city_size:]])
    test_city_list = [tc + '-test' for tc in test_city_list]
    train_city_list = list(np.array(city_list)[sort_index[:-1 * test_city_size]])
    return test_city_list, train_city_list
