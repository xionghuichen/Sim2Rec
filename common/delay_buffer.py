from common.config import *
from common.utils import *
from collections import deque


class DelayBuffer(object):
    def __init__(self, features_set, has_driver_dim, more_delay):
        self.features_set = features_set
        self.has_driver_dim = has_driver_dim
        self.feature_buffer = deque(maxlen=FeatureInfo.DRIVER_INFO_DELAY + more_delay)
        self.whether_buffer = deque(maxlen=FeatureInfo.WHETHER_DELAY + more_delay)
        self.whether_index = prefix_search(self.features_set, FeatureInfo.WHETHER_PREFIX)
        self.whether_index.extend(prefix_search(self.features_set, FeatureInfo.DAY_INFO_PREFIX))

        self.max_delay = max(FeatureInfo.DRIVER_INFO_DELAY, FeatureInfo.WHETHER_DELAY) + more_delay
        self.pop_time_step = deque(maxlen=self.max_delay + more_delay)

    def init_buf(self):
        # construct delay buffer
        self.time_step = 0
        # for i in range(self.max_delay):
        #     state = real_data[i]
        #     self.append(state)

    def append(self, state):
        state = np.asarray(state)
        self.feature_buffer.appendleft(state)
        self.pop_time_step.appendleft(self.time_step)
        self.time_step += 1
        if self.has_driver_dim:
            self.whether_buffer.appendleft(state[:, self.whether_index])
        else:
            self.whether_buffer.appendleft(state[self.whether_index])

    def pop(self):
        features = self.feature_buffer.pop()
        whether = self.whether_buffer.pop()
        ts = self.pop_time_step.pop()
        if self.has_driver_dim:
            features[:, self.whether_index] = whether
        else:
            features[self.whether_index] = whether
        return features, ts

    def get_next_state(self):
        features = self.feature_buffer[-1]
        whether = self.whether_buffer[-1]
        if self.has_driver_dim:
            features[:, self.whether_index] = whether
        else:
            features[self.whether_index] = whether
        return features

    def get_last_state(self, send_whether=True):
        features = self.feature_buffer[0]
        if send_whether and len(self.whether_buffer) > 0:
            whether = self.whether_buffer[0]
            if self.has_driver_dim:
                features[:, self.whether_index] = whether
            else:
                features[self.whether_index] = whether
        return features