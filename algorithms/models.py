import torch as th
import torch.nn as nn
import copy
from nets import NNModel, make_net
from controllers import Controller
from math import pi

class ActorNet(object):
    def __init__(self, env, hidden_layers=[], log_std_noise=-2): #, *args, **kwargs):
        self.net = make_net(
            [env.observation_space.shape[0]] + hidden_layers + [env.action_space.shape[0]],
            [nn.ReLU() for _ in hidden_layers]
        )
        # NNModel.__init__(self, net, *args, **kwargs)
        # super().__init__(env)
        self.log_std = th.FloatTensor([log_std_noise])
    
    def forward(self, X):
        return self.net.forward(X)
    
    def get_prob(self, action, state):
        norm_factor = 2 * pi * th.ones(action.shape[:-1])
        norm_factor = norm_factor.pow(action.shape[-1])
        return self.get_nlog_prob(action, state).exp() / norm_factor
    
    def get_nlog_prob(self, action, state):
        mu = self.forward(state)
        ret = (action - mu) / (self.log_std.exp())
        ret = -0.5 * ret.pow(2)
        return ret.sum(dim=-1) - self.log_std.expand_as(mu).sum(dim=-1)
    
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