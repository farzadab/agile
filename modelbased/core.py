# -*- coding: utf-8 -*-

# Python2 Compatibility
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

import torch as th
import torch.nn as nn

from nets import NNModel, make_net
from datastore.mem import ReplayMemory
from .math import log_avg_exp


class MPC_FVI(object):
    def __init__(self,
                 env,                          # the environment
                 hidden_size=32,               # neural network hidden layers size
                 ensemble_size=5,              # number of nets to use for the ensemble
                 nb_updates=10,                # number of optimizer updates for the dynamics and the value functions
                 batch_size=128,               # mini_batch_size for the optimizer
                 gamma=0.99, gae_lambda=0.95   # discount factor (γ) and the GAE lambda factor (λ)
                ):
        self.env = env
        self.hidden_size = hidden_size
        self.vfuns = self.make_value_nets(ensemble_size)
        self.nb_updates = nb_updates
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.memory = ReplayMemory(gamma, gae_lambda, 2*1000*1000)  # max size: 2M

    @property
    def sdim(self):
        return self.observation_space.shape[0]

    @property
    def adim(self):
        self.action_space.shape[0]
    
    def vhat(self, state):
        return log_avg_exp([v(state).detach() for v in self.vfuns])

    def make_value_nets(self, nb_nets):
        return [NNModel(
                net=make_net(
                    [self.sdim, self.hidden_size, self.hidden_size, self.hidden_size, 1],
                    [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]
                ),
                log_name='V_%d' % i,
            ) for i in range(nb_nets)]
    
    def get_mpc_action(state):
        pass  # TODO: MPC ......
    
    def update_dynamics(self, memory):
        # for _ in range(self.nb_updates):
        # use self.batch_size
        pass  # TODO: update f

    def update_critic(self, memory):
        # for _ in range(self.nb_updates):
        # use self.batch_size
        pass  # TODO: update vhat

    
    def polo(self, num_states, update_period):
        done = True
        memory = None
        for t in range(num_states):
            if done is True:
                state = env.reset()
            if memory is None:
                memory = ReplayMemory(self.gamma, self.gae_lambda)

            # choose action using MPC (TODO: num_particles)
            action = self.get_mpc_action(state)
            
            nstate, reward, done, _ = self.env.step(action)
            self.memory.record(state, action, reward, nstate, done=done)
            state = nstate

            if (t+1) % update_period == 0:
                memory.calculate_advantages(self.vhat)
                self.memory.extend(memory)
                self.update_dynamics(self.memory)
                self.update_critic(self.memory)







### TODO
# def main():
#     env = NormalizedEnv(
#         SerializableEnv(
#             name=args.env,
#             multi_step=args.multi_step,
#             randomize_goal=args.env_randomize_goal,
#             max_steps=args.env_max_steps,
#             reward_style=args.env_reward_style,
#             writer=writer,
#         ),
#         normalize_obs=True,
#         gamma=args.gamma,
#     )
#     env.init_normalization(1000)