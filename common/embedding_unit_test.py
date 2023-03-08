from common import logger


class EmbeddingUnitTest(object):
    FREQUENT = 'frequent_times'
    SINGLE = 'single_times'


    def __init__(self):
        self.func_dict = {}
        self.pass_test = {}
        self.test_type = {}
        self.test_param = {}
        pass

    def configure(self):
        pass

    def import_unit_test(self, name, func, type, **kwargs):
        self.func_dict[name] = func
        self.pass_test[name] = False
        self.test_type[name] = type
        self.test_param[name] = kwargs
        self.unit_test_init(name)

    def unit_test_init(self, name):
        if self.test_type[name] == EmbeddingUnitTest.FREQUENT:
            assert 'freq' in self.test_param[name]
            self.test_param[name]['times'] = 0

    def __unit_test(self, name, *args, **kwargs):
        pass_test, info = self.func_dict[name](*args, **kwargs)
        if not pass_test:
            raise Exception("error in : {}, info {}".format(name, info))
        else:
            logger.info("[pass unit test] {}".format(name))
        return pass_test, info

    def call_test(self, name, *args, **kwargs):
        if self.pass_test[name]:
            return
        else:
            if self.test_type[name] == EmbeddingUnitTest.SINGLE:
                pass_test, info = self.__unit_test(name, *args, **kwargs)
                if pass_test:
                    self.pass_test[name] = True
            elif self.test_type[name] == EmbeddingUnitTest.FREQUENT:
                times = self.test_param[name]['times']
                if times % self.test_param[name]['freq'] == 0:
                    pass_test, info = self.__unit_test(name, *args, **kwargs)
                    self.test_param[name]['times'] += 1

unittest = EmbeddingUnitTest()