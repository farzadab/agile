import torch as th
import numpy as np
import collections
import gym

from core.object_utils import ObjectWrapper


# Taken from https://gitlab.com/zxieaa/cassie.git
class Stats():
    def __init__(self, input_size, shift_mean=True, scale=True, clip=True):
        '''
        @brief used for normalizing a quantity (1D)
        @param input_size: the input_sizeension of the quantity
        '''
        # not really using the shared memory for now, but might be useful later
        self.shift_mean = shift_mean
        self.scale = scale
        self.clip = clip
        self.input_size = input_size
        self.reset()

    def observe(self, obs):
        '''
        @brief update observation mean & stdev
        @param obs: the observation. assuming flat and any dimension of size 1 will be removed first
        '''
        obs = th.FloatTensor(obs).squeeze()
        self.n += 1.
        self.sum = self.sum + obs
        self.sum_sqr += obs.pow(2)
        self.mean = self.sum / self.n
        self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-4,1e10).sqrt()
        self.mean = self.mean.float()
        self.std = self.std.float()

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = self.std
        inputs = th.FloatTensor(inputs)
        if len(inputs.shape) > 1:
            obs_mean = obs_mean.unsqueeze(0).expand_as(inputs)
            obs_std = obs_std.unsqueeze(0).expand_as(inputs)
        normalized = inputs
        if self.shift_mean:
            normalized -= obs_mean
        if self.scale:
            normalized /= obs_std
        if self.clip:
            normalized = th.clamp(normalized, -10.0, 10.0)
        # normalized = (inputs - obs_mean) / obs_std
        #obs_std = th.sqrt(self.var).unsqueeze(0).expand_as(inputs)
        return normalized

    def reset(self):
        self.n = th.zeros(self.input_size).share_memory_()
        self.mean = th.zeros(self.input_size).share_memory_()
        self.std = th.ones(self.input_size).share_memory_()
        self.sum = th.zeros(self.input_size).share_memory_()
        self.sum_sqr = th.zeros(self.input_size).share_memory_()


# originally from https://github.com/rll/rllab/tree/master/rllab/envs/normalized_env.py
class NormalizedEnv(ObjectWrapper):
    def __init__(
            self,
            env,
            normalize_obs=False,
            gamma=0,  # only used for normalization
    ):
        '''
        Assumes the observation and action spaces are always gym.spaces.Box(dtype=np.float32)

        @param normalize_obs: whether or not to normalize the observations
        @param gamma: the discount factor (γ), used to normalized the rewards. Default: no normalization
        '''
        super().__init__(env)
        self.gamma = gamma
        self.norm_stt = Stats(
            input_size=env.observation_space.shape[0],
            shift_mean=normalize_obs, scale=normalize_obs, clip=normalize_obs
        )  # won't normalize if `normalize_obs=False`
        # self.norm_rew = Stats(
        #     input_size=1,
        #     shift_mean=False, clip=False,  # never shift the reward
        #     scale=normalize_reward  # won't normalize if `normalize_reward=False`
        # )
        ub = np.ones(self.__wrapped__.action_space.shape)
        self.action_space = gym.spaces.Box(-1 * ub, ub, dtype=np.float32)

    # def _apply_normalize_obs(self, obs):
    #     self._update_obs_estimate(obs)
    #     return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    # def _apply_normalize_reward(self, reward):
    #     self._update_reward_estimate(reward)
    #     return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        stt = self.__wrapped__.reset()
        self.norm_stt.observe(stt)
        return self.norm_stt.normalize(stt)
    
    def step(self, action):
        # rescale the action
        ub = self.__wrapped__.action_space.high
        lb = self.__wrapped__.action_space.low
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        
        stt, rew, done, info = self.__wrapped__.step(scaled_action)

        self.norm_stt.observe(stt)

        return as_named_tuple(
            self.norm_stt.normalize(stt),
            rew * (1 - self.gamma), # self.norm_rew.normalize(rew),
            done,
            **info)

    def __str__(self):
        return "Normalized: %s" % self.__wrapped__

    # def log_diagnostics(self, paths):
    #     print "Obs mean:", self._obs_mean
    #     print "Obs std:", np.sqrt(self._obs_var)
    #     print "Reward mean:", self._reward_mean
    #     print "Reward std:", np.sqrt(self._reward_var)

_Step = collections.namedtuple("Step", ["state", "reward", "done", "info"])

# originally from https://github.com/rll/rllab/tree/master/rllab/envs/base.py
def as_named_tuple(state, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(state, reward, done, kwargs)
