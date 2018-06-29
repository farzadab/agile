import torch as th
import torch.nn as nn
import copy
from nets import NNModel, make_net
from controllers import Controller
from math import pi
import random

class ActorNet(object):
    def __init__(self, env, hidden_layers=[], log_std_noise=-2): #, *args, **kwargs):
        self.net = make_net(
            [env.observation_space.shape[0]] + hidden_layers + [env.action_space.shape[0]],
            [nn.ReLU() for _ in hidden_layers]
        )
        # import ipdb
        # ipdb.set_trace()  
        self.net[-1].state_dict()['bias'][:] = 0
        self.net[-1].state_dict()['weight'][:] /= 100
        # self.net[-1].state_dict()['weight'][:] = th.FloatTensor([[-.5, 0, .5, 0, 0, 0], [0, -.5, 0, .5, 0, 0]])
        # def pp(x):
        #     if random.random() < 0.005:
        #         print('grad:', x)
        # list(self.net[-1].parameters())[0].register_hook(pp)
        # NNModel.__init__(self, net, *args, **kwargs)
        # super().__init__(env)
        self.log_std = th.FloatTensor([log_std_noise])
    
    def forward(self, X):
        return self.net.forward(X)
    
    def set_noise(self, log_std):
        self.log_std = th.FloatTensor([log_std])
    
    def get_prob(self, action, state):
        norm_factor = 2 * pi * th.ones(action.shape[:-1])
        norm_factor = norm_factor.pow(action.shape[-1])
        return self.get_nlog_prob(action, state).exp() / norm_factor
    
    def get_nlog_prob(self, action, state):
        '''
        Computes the normalized log probability of taking actions under the current policy: log π(a|s) + const
        '''
        mu = self.forward(state)
        ret = (action - mu.expand_as(action)) / (self.log_std.exp().expand_as(action))
        ret = -0.5 * ret.pow(2)
        return ret.sum(dim=-1) - self.log_std.expand_as(action).sum(dim=-1)
    
    def get_action(self, state, explore=False):
        action = self.forward(state)
        if explore:
            action += self.log_std.exp() * th.randn_like(action)
        return action
    
    def zero_grad(self):
        self.net.zero_grad()
    
    def make_copy(self):
        c = copy.deepcopy(self)
        # TODO fix optimizer if needed
        return c
    
    def save_model(self, path):
        th.save(self.net, path)
    
    def load_model(self, path):
        self.net.load_state_dict(th.load(path))
    
    def is_linear(self):
        return len(self.net) == 1