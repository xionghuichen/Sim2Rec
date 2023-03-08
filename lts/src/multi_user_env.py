import sys
sys.path.append('../../')

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from lts.src import environment
from lts.src import recsim_gym
from lts.src import long_term_satisfaction
import gin.tf
from common import logger
from lts.src.config import *

# action space
class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, clickbait_score):
        self.clickbait_score = clickbait_score
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return np.array([self.clickbait_score])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Document {} with clickbait_score {}.".format(self._doc_id, self.clickbait_score)



# action space for each time-step
class LTSDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_size, doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0.
        self.doc_size = doc_size

    def sample_document(self):
        doc_features = {}
        doc_features['doc_id'] = int(self._doc_count)
        doc_features['clickbait_score'] = self._doc_count / self.doc_size
        self._doc_count += 1
        return self._doc_ctor(**doc_features)  # param -> doc obj



class LTSUserState(user.AbstractUserState):
  """Class to represent users.

  See the LTSUserModel class documentation for precise information about how the
  parameters influence user dynamics.
  Attributes:
    memory_discount: rate of forgetting of latent state.
    sensitivity: magnitude of the dependence between latent state and
      engagement.
    innovation_stddev: noise standard deviation in latent state transitions.
    choc_mean: mean of engagement with clickbaity content.
    choc_stddev: standard deviation of engagement with clickbaity content.
    kale_mean: mean of engagement with non-clickbaity content.
    kale_stddev: standard deviation of engagement with non-clickbaity content.
    net_positive_exposure: starting value for NPE (NPE_0).
    time_budget: length of a user session.
  """
  given_ep = False
  given_gp = False
  def __init__(self, memory_discount, sensitivity, innovation_stddev,
               choc_mean, choc_stddev, kale_mean, kale_stddev,
               net_positive_exposure, time_budget, std_level, sim_noise_level,
              ):
    """Initializes a new user."""
    ## Transition model parameters
    ##############################
    self.memory_discount = memory_discount
    self.sensitivity = sensitivity
    self.innovation_stddev = innovation_stddev

    ## Engagement parameters
    self.choc_mean = choc_mean
    self.choc_stddev = choc_stddev
    self.kale_mean = np.clip(kale_mean + np.random.uniform(-1 * sim_noise_level, sim_noise_level), 0, 100)
    self.kale_stddev = kale_stddev
    # 以上是环境特征， 以下是真实状态变量， 若前者已知，下面的可以由 observation 输出。

    ## State variables
    ##############################
    self.net_positive_exposure = net_positive_exposure
    self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_positive_exposure))
    self.time_budget = time_budget
    self.gp_state = np.random.normal(self.choc_mean, std_level)
    self.hd_list = ['memory_discount', 'sensitivity', 'satisfication']

  def create_observation(self):
      # TODO: 该部分可以设置部分可知，取决于具体我们的实验设置
    """User's state is not observable."""
    if self.given_ep:
        return np.array([self.memory_discount, self.sensitivity, self.satisfaction])
    elif self.given_gp:
        return np.array([self.gp_state])
        # return np.array([self.choc_mean])
    else:
      return np.array([])

  def create_hidden_state(self):
      return np.array([self.memory_discount, self.sensitivity, self.satisfaction])

  # No choice model.
  def score_document(self, doc_obs):
    return 1

  @classmethod
  def observation_space(cls):
    if cls.given_ep:
        return spaces.Box(dtype=np.float32, low=np.zeros(3), high=np.ones(3) * np.inf)
    elif cls.given_gp:
        return spaces.Box(dtype=np.float32, low=np.zeros(1), high=np.ones(1) * np.inf)
    else:
      return spaces.Box(shape=(0,), dtype=np.float32, low=0.0, high=np.inf)


@gin.configurable
# TODO： 多用户多domain，这里要重新修改
class LTSUserSampler(user.AbstractUserSampler):
  """Generates user with identical predetermined parameters."""
  _state_parameters = None
  def __init__(self, user_ctor=LTSUserState,
               # 转移特征，每一个环境不一样
               # 下面两项决定了一个动作的影响的程度，是转移的参数
               # memory_discount=0.7,
               # sensitivity=0.01,
               innovation_stddev=0, # 0.05,
               # 环境特征，每一个domain不一样，相同domain可以一样
               choc_mean=5.0, choc_stddev=0.01,
               kale_mean=4.0, kale_stddev=0.01,
               time_budget=60,
               given_ep=False, given_gp=False,
               log_sample=False, std_level=1.0,
               sim_noise_level=0.0,
               **kwargs):
    """Creates a new user state sampler."""
    logger.debug('Initialized LTSStaticUserSampler')
    if given_ep:
        LTSUserState.given_ep = True
    else:
        LTSUserState.given_ep = False
    if given_gp:
        LTSUserState.given_gp = True
    else:
        LTSUserState.given_gp = False
    if not log_sample:
        self._state_parameters = {'memory_discount': np.random.uniform() * (MEMORY_DISCOUNT_RANGE[1] - MEMORY_DISCOUNT_RANGE[0]) +
                                                     MEMORY_DISCOUNT_RANGE[0], # memory_discount,
                                  'sensitivity': np.random.uniform() * (SENSITIVITY_RANGE[1] - SENSITIVITY_RANGE[0]) + SENSITIVITY_RANGE[0],
                                  'innovation_stddev': innovation_stddev,
                                  'choc_mean': choc_mean,
                                  'choc_stddev': choc_stddev,
                                  'kale_mean': kale_mean,
                                  'kale_stddev': kale_stddev,
                                  'time_budget': time_budget,
                                  "std_level": std_level,
                                  "sim_noise_level": sim_noise_level}
    else:
        self._state_parameters = {'memory_discount': np.log(np.random.uniform() * (LOG_MEMORY_DISCOUNT_RANGE[1] - LOG_MEMORY_DISCOUNT_RANGE[0]) + LOG_MEMORY_DISCOUNT_RANGE[0]), # memory_discount,
                                  'sensitivity': np.log(np.random.uniform() * (LOG_SENSITIVITY_RANGE[1] - LOG_SENSITIVITY_RANGE[0]) + LOG_SENSITIVITY_RANGE[0]),
                                  'innovation_stddev': innovation_stddev,
                                  'choc_mean': choc_mean,
                                  'choc_stddev': choc_stddev,
                                  'kale_mean': kale_mean,
                                  'kale_stddev': kale_stddev,
                                  'time_budget': time_budget,
                                  "std_level": std_level,
                                  "sim_noise_level": sim_noise_level,
                                 }
    super(LTSUserSampler, self).__init__(user_ctor, **kwargs)

  def sample_user(self):
    # _state_parameters 是不变量，而每次重新sample的时候，以下变量会发生改变
    # starting_npe = ((self._rng.random_sample() - .5) *
    #                 (1 / (1.0 - self._state_parameters['memory_discount'])))
    starting_npe = (0.5 *
                    (1 / (1.0 - self._state_parameters['memory_discount'])))
    self._state_parameters['net_positive_exposure'] = starting_npe
    return self._user_ctor(**self._state_parameters)


class LTSResponse(user.AbstractResponse):
  """Class to represent a user's response to a document.

  Attributes:
    engagement: real number representing the degree of engagement with a
      document (e.g. watch time).
    clicked: boolean indicating whether the item was clicked or not.
  """

  # The maximum degree of engagement.
  MAX_ENGAGEMENT_MAGNITUDE = 100.0

  def __init__(self, clicked=False, engagement=0.0):
    """Creates a new user response for a document.

    Args:
      clicked: boolean indicating whether the item was clicked or not.
      engagement: real number representing the degree of engagement with a
        document (e.g. watch time).
    """
    self.clicked = clicked
    self.engagement = engagement

  def __str__(self):
    return '[{}]'.format(self.engagement)

  def __repr__(self):
    return self.__str__()

  def create_observation(self):
    return [int(self.clicked), np.clip(self.engagement, 0, LTSResponse.MAX_ENGAGEMENT_MAGNITUDE)]

  @classmethod
  def response_space(cls):
    # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
    low = np.array([0, 0])
    high = np.array([1, LTSResponse.MAX_ENGAGEMENT_MAGNITUDE])
    return spaces.Box(
                low=low,
                high=high,
                dtype=np.float32)

def make(num_users, domain_name, doc_size=30, choc_mean=5.0, kale_mean=4.0, time_budget=60,
         cgc_type=-1, given_ep=False, given_gp=False, log_sample=False, std_level=1.0, sim_noise_level=0.0):
    slate_size = 1
    user_models = []
    for i in range(num_users):
        user_model = long_term_satisfaction.LTSUserModel(slate_size=1,
                                                        user_state_sampler=lambda :LTSUserSampler(choc_mean=choc_mean,
                                                                                                  kale_mean=kale_mean,
                                                                                                  time_budget=time_budget,
                                                                                                  given_ep=given_ep,
                                                                                                  given_gp=given_gp,
                                                                                                  log_sample=log_sample,
                                                                                                  std_level=std_level,
                                                                                                  sim_noise_level=sim_noise_level),
                                                        response_model_ctor=LTSResponse, # 响应空间，是observation的一部分
                                                         )
        user_models.append(user_model)
    num_candidates = doc_size

    """Class to represent environment with multiple users.
    
    Attributes:
      user_model: A list of AbstractUserModel instances that represent users.
      document_sampler: An instantiation of AbstractDocumentSampler.
      num_candidates: An integer representing the size of the candidate_set.
      slate_size: An integer representing the slate size.
      num_clusters: An integer representing the number of document clusters.
    """

    ltsenv = environment.MultiUserEnvironment(
        user_models,
        LTSDocumentSampler(doc_size),
        num_candidates,
        slate_size,
        resample_documents=False)


    def clicked_engagement_reward(responses):
        rewards = []
        for response in responses:
            reward = 0
            assert len(response) == 1
            response = response[0]
            if response.clicked:
                reward += response.engagement
            rewards.append(reward)
        return rewards
    lts_gym_env = recsim_gym.RecSimGymEnvFlat(raw_environment=ltsenv, reward_aggregator=clicked_engagement_reward)
    lts_gym_env_vec = recsim_gym.RecSimGymEnvVec(lts_gym_env, domain_name=domain_name, cgc_type=cgc_type, log_sample=log_sample)
    return lts_gym_env_vec


if __name__ == '__main__':
    doc_size = 30
    num_users = 10
    lts_gym_env = make(num_users, doc_size=doc_size)
    observation_0 = lts_gym_env.reset()
    print('Observation 0 {}'.format(observation_0))

    # Agent recommends the first three documents.， 因为是连续控制，我们直接改成doc 的kaleness 值，那么需要修改 document 的sampler
    recommendation_slate_0 = [[int(0.23 * doc_size)] for i in range(num_users)]
    observation_1, reward, done, info = lts_gym_env.step(recommendation_slate_0)
    print('Observation 1 {}'.format(observation_1))
    print("hidden state {}".format(info['hidden_state']))
