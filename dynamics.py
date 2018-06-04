import torch as th
import torch.nn as nn
import torch.optim as optim
import random

from nets import NNModel, make_net
from logs import logger


def dtensor(v):
    return th.DoubleTensor(v)

class Data(dict):
    key_names = ['s', 'a', 's_n']
    def __init__(self, *args):
        super(Data, self).__init__([(k, args[i] if len(args) > i else []) for i,k in enumerate(self.key_names)])
        self._size = 0 if len(args) == 0 else len(args[0])

    def size(self):
        '''The same as `len`, but trying not to mess with the `dict` structure'''
        return self._size

    def add_point(self, *args):
        for i, k in enumerate(self.key_names):
            self[k].append(args[i])
        self._size += 1

    def extend(self, other):
        for k in self.key_names:
            self[k].extend(other[k])
        self._size += other.size()

    def sample(self, sample_size=None):
        if sample_size is None or sample_size > self.size():
            return self
        samples = random.sample(range(self.size()), min(sample_size, self.size()))

        return [
            [self[k][i] for i in samples]
            for k in self.key_names
        ]  # state, action, next_state
    
    def get_all(self):
        return [ self[k] for k in self.key_names ]  # state, action, next_state


class Normalization(object):
    '''
        Internal class of DynamicsModel used for normalization
    '''
    def __init__(self, values, keys):
        self.stats = {}
        for k in keys:
            self.stats[k] = {
                'mean': th.mean(dtensor(values[k]), 0),
                'std' : th.std (dtensor(values[k]), 0) + 1e-6,
            }
    
    def normalize(self, value, key):
        return (dtensor(value) - self.stats[key]['mean']) / self.stats[key]['std']

    def unnormalize(self, value, key):
        return dtensor(value) * self.stats[key]['std'] + self.stats[key]['mean']
        


class DynamicsModel(NNModel):
    def __init__(self, env, net, init_data):
        '''
            params:
                env: used to get some high-level info about environment (i.e. observation_space)
                net: the neural network to be used
                init_data: used for normalization
        '''
        self.env = env
        super().__init__(net)
        self.multipliers = th.ones(env.observation_space.shape[0], dtype=th.double)
        # self.multipliers *= 10
        # self.multipliers[3] = 20
        # TODO: check see if we need to normalize state and next_state separately or not: just concat s and s_n keys
        self.norm = Normalization(
            {'s': init_data[0] + init_data[2],
             'a': init_data[1]},
            ['s', 'a']
        )
    
    def fit(self, states, actions, next_states):
        states = self.norm.normalize(states, 's')
        actions = self.norm.normalize(actions, 'a')
        next_states = self.norm.normalize(next_states, 's')
        return super().fit(
            th.cat([states, actions], 1),
            (next_states - states) * self.multipliers
        )
    
    def predict(self, state, action):
        '''
            Assuming just one 
        '''
        state = self.norm.normalize(state, 's')
        action = self.norm.normalize(action, 'a')
        return self.norm.unnormalize(
            super().predict(th.cat([state, action])) / self.multipliers + state,
            's'
        )
