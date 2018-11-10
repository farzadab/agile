# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

import torch as th
import torch.nn as nn

from nets import NNModel, make_net


class DynamicsModel(NNModel):
    def __init__(self, net=None, env=None, hidden_size=32, *args, **kwargs):
        if net is None and env is None:
            raise ValueError('DynamicsModel init: One of `net` and `env` should be provided.')
        if net is None:
            adim = env.action_space.shape[0]
            sdim = env.observation_space.shape[0]
            net = make_net(
                [sdim+adim, hidden_size, hidden_size, hidden_size, sdim],
                [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]
            )
        super(DynamicsModel, self).__init__(*args, **kwargs)
    
    def fit_traj_batch(self, states, actions, nstates, dones):
        self.net.train()

        avg_loss = 0

        # zero the parameter gradients
        self.optimizer.zero_grad()

        nb_samples = states.shape[0] * states.shape[1]

        for i in range(states.shape[0]):
            if i == 0:
                s = states[0]
            else:
                # propagate if not done:
                s = s * (1-dones[i-1]) + states[i] * dones[i-1]

            # forward + backward + optimize
            inp = th.cat((s, actions[i]), dim=1)
            pred = self.net.forward(inp)
            loss = self.criterion(pred, nstates[i])
            loss.backward()
            self.optimizer.step()
            avg_loss += float(loss) / nb_samples
        
        # write summary to PyTorch-Tensorboard
        self.i_iteration += 1
        if self.writer:
            self.writer.add_scalar(self.log_label, loss, self.i_iteration)
            
        return loss

    def fit_traj(self, traj, horizon=16, batch_size=128):
        num_samples = batch_size / horizon
        sample_start_inds = th.randperm(traj.size())[:num_samples]
        sample_inds = th.stack([sample_start_inds+i for i in range(horizon)])
        state = traj['state'][sample_inds]
        action = traj['action'][sample_inds]
        nstate = traj['nstate'][sample_inds]
        done = traj['done'][sample_inds]
        self.fit_traj_batch(state, action, nstate, done)
