DAY_PREDICT = 1
WEATHER_DELAY = 2
HIST_LOOK_BACK = 7
HIST_PULL_DELAY = 1

class FeatureInfo(object):
    DRIVER_INFO_DELAY = 2
    WHETHER_DELAY = 1
    WHETHER_PREFIX = 'caiyun'
    DAY_INFO_PREFIX = 'date_type'

class EvaluationType(object):
    NO = 'no'
    EVALUATION = 'eval'
    COST_CONSTRAINT = 'cost'

class Optimizer(object):
    ADAM = 'adam'
    RADAM = 'radam'

class DynamicType(object):
    NULL_COUPON = 'null_coupon'
    POLICY = 'policy'
    EXPERT = 'expert_policy'
    EXPERT_DATA = 'expert_data'
    OPTIMAL = 'optimal'
    SUB_OPTIMAL = 'sub_optimal'
    BIGGEST_COUPON_POLICY = 'bcp'

class AlgorithmType(object):
    TRPO = 'trpo'
    DFM = 'dfm'
    PPO = 'ppo'

class ClusterType(object):
    FOS = 'fos'
    RAND = 'rand'
    MAP = 'map'


class HandCodeType(object):
    MAX_TRANS = 'max'
    STD_TRANS = 'std'

class TransType(object):
    DIRECT = 'direct'
    DIRECT_TRANS = 'direct_trans'
    DIRECT_TRANS_CITY = 'direct_trans_city'
    UNIVERSAL = 'universal'
    ENV_AWARE = 'env_aware'
    ENV_AWARE_VAN = 'env_aware_van'
    ENV_AWARE_DIRECT = 'env_aware_dir'
    BCQ = 'bcq'

class ActFun(object):
    TANH = 'tanh'
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'

    @classmethod
    def gen_act(cls, config):
        import tensorflow as tf
        if config == cls.TANH:
            return tf.nn.tanh
        elif config == cls.RELU:
            return tf.nn.relu
        elif config == cls.LEAKY_RELU:
            return tf.nn.leaky_relu
        else:
            raise NotImplementedError

class ConstraintLDA(object):
    NO = 'no'
    Proj2CGC = 'pcgc'
    Proj2PG ='ppg'

class TrajInfo(object):
    OBS = 'obs'
    ACS = 'acs'
    CONSTRAINT = 'constraint'
    SCALE_PARAM = 'scale_param'

class ConstraintType(object):
    NUMBER = 5
    MAX_STANDARD_FOS = 'max_acs'
    MEAN_STANDARD_FOS = 'mean_acs'
    STD_STANDARD_FOS = 'std_acs'
    NO_ZERO_FOS_RATIO = 'no_zero_fos_ratio'
    COUPON_SENSITIVE_RATIO = 'coupon_sensitive_ratio'
    ORDER = [MAX_STANDARD_FOS, MEAN_STANDARD_FOS, STD_STANDARD_FOS, NO_ZERO_FOS_RATIO, COUPON_SENSITIVE_RATIO]

class ClusterFeatures(object):
    ORDER = [ConstraintType.MAX_STANDARD_FOS, ConstraintType.MEAN_STANDARD_FOS, ConstraintType.STD_STANDARD_FOS, ConstraintType.NO_ZERO_FOS_RATIO]


class ExpertStatistics(object):
    COUPON_RATIO = 'coupon_ratio'
    COST = 'cost'

class VaeTaskType(object):
    STATIC = 'static'
    FOS = 'fos'
    COUPON = 'coupon'
    STATIC_FOS = 's_f'
    STATIC_COUPON = 's_c'
    ALL = 'all'


class VaeDataType(object):
    CAT = 'cat'
    NORM = 'norm'
    ALL = 'all'

# unit test type
class UnitTestType(object):
    NULL = 'null'
    SUB_OPTIMAL_TEST = 'sub_optimal_test'
