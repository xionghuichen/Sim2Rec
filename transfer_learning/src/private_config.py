city_list = ['Weifang', 'Huhehaote', 'Meizhou', 'Heyuan']
test_city_list = ['Meizhou']
train_city_list = ['Weifang'] # , 'Huhehaote']
DATA_SET_ROOT = '../../feature_engineering/dataset/'
LOG_ROOT = '../'
CHECKPOINT_ROOT = '../../'

vae_checkpoint_path = CHECKPOINT_ROOT + 'vae/checkpoint/'
vae_checkpoint_date = '2020/01/04/19-48-11-818403'  # '2019/11/16/10-52-43-111222'

demer_checkpoint_path = CHECKPOINT_ROOT + 'model/checkpoint/'
demer_test_checkpoint_date = demer_checkpoint_date = [
                        '2019/09/28/22-01-01-059227',
                         '2019/09/28/21-56-24-323785',
                         '2019/09/28/21-57-14-788876',
                         '2019/09/28/21-56-49-050952'
]
demer_date_range = [
                    [20190601, 20190630],
                    [20190601, 20190630],
                    [20190601, 20190630],
                    [20190601, 20190630],
                    ]

demer_test_date_range = [
                    [20190701, 20190730],
                    [20190701, 20190730],
                    [20190701, 20190730],
                    [20190701, 20190730],
                    ]

demer_folder_name = [
                'Weifang_ceil_reduce_single_day_coupon_29',
                'Huhehaote_ceil_reduce_single_day_coupon_29',
                'Meizhou_ceil_reduce_single_day_coupon_29',
                'Heyuan_ceil_reduce_single_day_coupon_29',

                ]

relation_matrix_file_name = 'relation_matrix-c_3-d_10'
