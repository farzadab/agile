from gym.utils import seeding
import numpy as np
import os.path as path
import copy
import time
import gym

import cust_envs

from algorithms.plot import ScatterPlot, QuiverPlot, Plot
from core.object_utils import ObjectWrapper
from algorithms.normalization import NormalizedEnv
from envs.paths import CircularPath, LineBFPath, DiscretePath, RPath1
from algorithms.senv import PointMass


class PMFollow(PointMass):
    max_speed = 4.
    # TODO: use arg to see if phase should be shown or act
    def __init__(self, path_gen=None, nb_lookaheads=2, start_at_goal=True, *args, **kwargs):
        self.parent_indices = [0,1,4,5]
        super().__init__(reset=False, *args, **kwargs)
        # self.phase_start = 0
        # self.phase_end   = 1
        self.phase = 0
        self.path_gen = path_gen if path_gen else RPGen()
        self.nb_lookaheads = nb_lookaheads
        high_la = self.max_position * np.ones(nb_lookaheads * 2)
        high = np.concatenate([self.observation_space.high[self.parent_indices], high_la])
        low  = np.concatenate([self.observation_space.low[self.parent_indices],  -high_la])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.start_at_goal = start_at_goal
        self.treshold = -1.  # should never achieve the goal, just needs to follow it
        self.reset()

    def reset(self):
        self.path = self.path_gen.gen_path()
        super().reset()
        self.phase = 0
        # if self.randomize_goal:
        #     self.phase = self.np_random.uniform(self.phase_start, self.phase_end)
        self._set_goal_pos()
        if self.start_at_goal:
            self.state[0:2] = self.state[2:4]
        return self._get_obs()

    def _set_goal_pos(self):
        self.state[2:4] = np.array(self.path.at_point(self.phase)) * self.max_position

    def step(self, u):
        self.phase += self.dt / self.path.duration()
        self._set_goal_pos()
        done = (self.phase >= 1)
        obs, rew, _, ext = super().step(u)
        return obs, rew, done, ext

    def _get_obs(self):
        # while self.phase > 1:
        #     self.phase -= 1
        # while self.phase < 0:
        #     self.phase += 1
        return np.concatenate(
            [
                self.state[self.parent_indices],
                self.max_position * np.concatenate([
                    self.path.at_point(self.phase + (i+1) * self.dt)
                    for i in range(self.nb_lookaheads)
                ])
            ])


class PGen(object):
    def gen_path(self):
        raise NotImplementedError


class RPGen(PGen):
    def __init__(self, nb_points=5, seed=None):
        self.nb_points = nb_points
        self.np_random, _ = gym.utils.seeding.np_random(seed)
    
    def gen_path(self):
        return DiscretePath(
            points=[
                self.np_random.uniform([-1, -1], [1,1])
                for _ in range(self.nb_points)
            ],
            smooth=True,
        )