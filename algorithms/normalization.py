import torch as th
import numpy as np

# Taken from https://gitlab.com/zxieaa/cassie.git
class Stats():
    def __init__(self, input_size):
        '''
        @brief used for normalizing a quantity (1D)
        @param input_size: the input_sizeension of the quantity
        '''
        # not really using the shared memory for now, but might be useful later
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
        self.std = (self.sum_sqr / self.n - self.mean.pow(2)).clamp(1e-2,1e9).sqrt()
        self.mean = self.mean.float()
        self.std = self.std.float()

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = self.std
        inputs = th.FloatTensor(inputs)
        if len(inputs.shape) > 1:
            obs_mean = obs_mean.unsqueeze(0).expand_as(inputs)
            obs_std = obs_std.unsqueeze(0).expand_as(inputs)
        normalized = (inputs - obs_mean) / obs_std
        #obs_std = th.sqrt(self.var).unsqueeze(0).expand_as(inputs)
        return th.clamp(normalized, -10.0, 10.0)

    def reset(self):
        self.n = th.zeros(self.input_size).share_memory_()
        self.mean = th.zeros(self.input_size).share_memory_()
        self.std = th.ones(self.input_size).share_memory_()
        self.sum = th.zeros(self.input_size).share_memory_()
        self.sum_sqr = th.zeros(self.input_size).share_memory_()