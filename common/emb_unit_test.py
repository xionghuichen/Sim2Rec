from common import logger
class BasicUnitTest(object):
    def __init__(self):
        pass

class TestType(object):
    SINGLE_TIME = 'single_time'
    FREQUENT = 'freq'
    HASH_MAP = 'hash'


class EmbUnitControler(BasicUnitTest):
    def __init__(self):
        BasicUnitTest.__init__(self)
        self.test_obj_dict = {}
        self.test_type = TestType

    def add_test(self, test_func_name, test_func, test_type, freq=-1, hash_keys=None):
        test_obj = EmbUnitTestObj(test_func, test_type, freq, hash_keys)
        if test_func_name in self.test_obj_dict:
            return
        else:
            self.test_obj_dict[test_func_name] = test_obj

    def do_test(self, func_name, hash_key=None, *args, **kwargs):
        assert func_name in self.test_obj_dict
        self.test_obj_dict[func_name](hash_key, func_name, *args, **kwargs)
    #
    # def build_test(self, func):
    #     if hasattr(self, str(id(func))):
    #         pass
    #     else:
    #         def wrap_func(*args, **kwargs):
    #             if self.__getattribute__(str(id(func))):
    #                 pass
    #             else:
    #                 should_pass = func(*args, **kwargs)
    #                 if should_pass:
    #                     self.__setattr__(str(id(func)), True)
    #                     print("[pass test]")
    #         return wrap_func
    #

class EmbUnitTestObj(object):

    def __init__(self, test_func, test_type, freq=-1, hash_keys=None):
        self.test_type = test_type
        self.name = test_func.__name__
        self.test_func = test_func
        self.__do_test_var = {}
        self.__do_test_var_init(freq, hash_keys)

    def __do_test_var_init(self, freq, hash_keys):
        if self.test_type == TestType.SINGLE_TIME:
            self.__do_test_var['res'] = True
        elif self.test_type == TestType.FREQUENT:
            assert type(freq) is int
            self.__do_test_var = {"count": 0, "freq": freq}
        elif self.test_type == TestType.HASH_MAP:
            assert hash_keys is not None
            for k in hash_keys:
                self.__do_test_var[k] = True

    def __if_do_test(self, hash_key=None):
        if self.test_type == TestType.SINGLE_TIME:
            if self.__do_test_var['res']:
                self.__do_test_var['res'] = False
                return True
            else:
                return False
        elif self.test_type == TestType.FREQUENT:
            assert type(self.__do_test_var) is dict
            self.__do_test_var['count'] += 1
            return self.__do_test_var['count'] % self.__do_test_var['freq'] == 0
        elif self.test_type == TestType.HASH_MAP:
            if not self.__do_test_var[hash_key]:
                self.__do_test_var[hash_key] = True
                return True
            else:
                return False

    def __call__(self, hash_key=None, func_name=None, *args, **kwargs):
        if self.__if_do_test(hash_key):
            logger.info("do test {}, {} - {}".format(self.test_type, self.name, func_name))
            self.test_func(*args, **kwargs)







emb_unit_test_cont = EmbUnitControler()
