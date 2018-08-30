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
    def __init__(self, path_gen=None, nb_lookaheads=2, goal_in_state=False, start_at_goal=True, *args, **kwargs):
        self.parent_indices = [0,1,4,5]
        self.goal_in_state = goal_in_state
        super().__init__(reset=False, *args, **kwargs)
        # self.phase_start = 0
        # self.phase_end   = 1
        self.phase = 0
        self.path_gen = path_gen if path_gen else RPGen()
        self.nb_lookaheads = nb_lookaheads
        high_la = self.max_position * np.ones((nb_lookaheads + goal_in_state) * 2)
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
                    self.path.at_point(self.phase + i * self.dt)
                    for i in range(1-self.goal_in_state, self.nb_lookaheads+1)
                ])
            ])
    
    def _get_geoms(self):
        from gym.envs.classic_control import rendering
        self.lookahead_points = [
            rendering.make_circle(1, filled=False)
            for _ in range(self.nb_lookaheads)
        ]
        self.lookahead_transforms = []
        for i, p in enumerate(self.lookahead_points):
            t = rendering.Transform()
            p.add_attr(t)
            p.set_color(1 * (self.nb_lookaheads - i), 0.1, 0.1)
            self.lookahead_transforms.append(t)
        return self.lookahead_points + super()._get_geoms()
    
    def _do_transforms(self):
        super()._do_transforms()
        for i, t in enumerate(self.lookahead_transforms):
            point = self.max_position * np.array(self.path.at_point(self.phase + (i+1) * self.dt))
            t.set_translation(point[0], point[1])


class PMFollow1(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=1, *args, **kwargs)


class PMFollow4(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=4, *args, **kwargs)


class PMFollow8(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=8, *args, **kwargs)


class PMFollowG1(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=1, goal_in_state=True, *args, **kwargs)

class PMFollowG2(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=2, goal_in_state=True, *args, **kwargs)

class PMFollowG4(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=4, goal_in_state=True, *args, **kwargs)

class PMFollowG8(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=8, goal_in_state=True, *args, **kwargs)


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


class PMFollowIce(PMFollow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def control_coeff(self, p, v, u):
        return 1
    def _integrate(self, p, v, u):
        return super()._integrate(
            p,
            v,
            u * self.control_coeff(p, v, u)
        )


class PMFollowIceMid(PMFollowIce):
    def control_coeff(self, p, v, u):
        if abs(p[1]) < self.max_position / 2:
            return 0.01
        return 1

    def _get_geoms(self):
        from gym.envs.classic_control import rendering
        ice_shape = [
            [-self.max_position, -self.max_position/2],
            [-self.max_position, self.max_position/2],
            [self.max_position, self.max_position/2],
            [self.max_position, -self.max_position/2],
        ]
        ice = rendering.make_polygon(v=ice_shape, filled=True)
        ice.set_color(0.65, 0.95, 0.96)
        return [ice] + super()._get_geoms()


class PMFollowIceMid1(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=1, *args, **kwargs)

class PMFollowIceMid4(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=4, *args, **kwargs)

class PMFollowIceMid8(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=8, *args, **kwargs)


class PMFollowGIceMid1(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=1, goal_in_state=True, *args, **kwargs)

class PMFollowGIceMid2(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=2, goal_in_state=True, *args, **kwargs)

class PMFollowGIceMid4(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=4, goal_in_state=True, *args, **kwargs)

class PMFollowGIceMid8(PMFollowIceMid):
    def __init__(self, *args, **kwargs):
        super().__init__(nb_lookaheads=8, goal_in_state=True, *args, **kwargs)