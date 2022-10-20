import numpy as np
import random
class CouponActionInfo(object):
    def __init__(self, coupon_mean_std, coupon_min_max):
        self.coupon_mean = coupon_mean_std[:, 0]
        self.coupon_std = coupon_mean_std[:, 1]
        self.coupon_min = coupon_min_max[:, 0]
        self.coupon_max = coupon_min_max[:, 1]
        self.precision = 100

    def configure(self, precision):
        self.precision = 10 ** precision

    def sample_action(self, size):
        c_level_1_threshold = np.random.randint(self.coupon_min[0], self.coupon_max[0] + 1, size=size)
        c_level_2_threshold = np.random.randint(self.coupon_min[2], self.coupon_max[2] + 1, size=size)
        c_level_1_unit_coupon = np.random.randint(self.coupon_min[1] * self.precision, self.coupon_max[1] * self.precision + 1, size=size) / self.precision
        c_level_2_unit_coupon = np.random.randint(self.coupon_min[3] * self.precision, self.coupon_max[3] * self.precision + 1, size=size) / self.precision
        levels = np.random.randint(0, self.coupon_max[2] + 1, size=size)
        x = np.stack([c_level_1_threshold, c_level_1_unit_coupon, c_level_2_threshold, c_level_2_unit_coupon, levels], axis=0).T
        return self.construct_coupon_info(x)

    def get_range(self):
        dim_range = [[int(self.coupon_min[0]), int(self.coupon_max[0]) + 1],
                     [int(self.coupon_min[1] * self.precision), int(self.coupon_max[1] * self.precision + 1)],
                     [int(self.coupon_min[2]), int(self.coupon_max[2] + 1)],
                     [int(self.coupon_min[3] * self.precision), int(self.coupon_max[3] * self.precision + 1)],
                     [0, int(self.coupon_max[2] + 1)]]
        dim_type = [False, False, False, False, False]
        return dim_range, dim_type

    def construct_coupon_info(self, x, sec_reshape=False, is_r_factor=False, rescale=True):
        c_level_1_threshold = x[:, 0]
        c_level_1_unit_coupon = x[:, 1]
        if sec_reshape:
            c_level_2_threshold = x[:, 2] - x[:, 0]
            c_level_2_unit_coupon = x[:, 3] - x[:, 1]
        else:
            c_level_2_threshold = x[:, 2]
            c_level_2_unit_coupon = x[:, 3]
        if is_r_factor:
            r = x[:, 4]
            levels = np.clip(np.round(1 / (1 - r + 0.001), 0).astype('int32'), 1, c_level_2_threshold) + 1
        else:
            levels = x[:, 4] % (c_level_2_threshold + 0.001) + 2
        r_factor = 1 - 1 / (levels - 1)
        coupon_info = np.stack(
            [c_level_1_threshold, c_level_1_unit_coupon, c_level_2_threshold, c_level_2_unit_coupon, r_factor], axis=1)
        coupon_info = np.clip(coupon_info, 0, None)
        if rescale:
            return (coupon_info - np.expand_dims(self.coupon_mean, axis=0)) / np.expand_dims(self.coupon_std, axis=0)
        else:
            return coupon_info


if __name__== '__main__':
    coupon_mean_std = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    coupon_min_max = np.array([[0, 20], [0, 2.5], [0, 10], [0, 19], [0, 1]])
    cai = CouponActionInfo(coupon_mean_std, coupon_min_max)
    cai.precision = 100
    print(cai.sample_action(10))
