from gym.envs.classic_control import rendering
from gym.utils import seeding
import numpy as np
import os.path as path
import gym

from algorithms.plot import ScatterPlot, QuiverPlot, Plot


class PointMass(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }

    def __init__(self, randomize_goal=True, writer=None):
        self.max_speed = 2.
        self.max_torque = 2.
        self.max_position = 20.
        self.treshold = 2.
        self.goal_reward = 100
        self.dt = .05
        self.mass = .2
        self.randomize_goal = randomize_goal

        self.writer = writer

        self.viewer = None
        self.plot = None

        # self.obs_size = 4
        self.act_size = 2
        
        high_action = self.max_torque * np.ones(self.act_size)
        self.action_space = gym.spaces.Box(low=-high_action, high=high_action)

        high_position = np.concatenate([self.max_position * np.ones(4), self.max_speed * np.ones(2)])
        self.observation_space = gym.spaces.Box(low=-high_position, high=high_position)

        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def step(self, u):
        # state: (px, py, gx, gy, vx, vy)
        p = self.state[0:2]
        g = self.state[2:4]
        v = self.state[4:6]
        
        reward = 0

        u = np.array(u)  # PyTorch poblems: fixing it from both sides

        if np.linalg.norm(u) > self.max_torque:
            u = u / np.linalg.norm(u) * self.max_torque
        
        self.last_u = u

        # just a simple (dumb) explicit integration ... 
        v = v + u * self.dt / self.mass
        if np.linalg.norm(v) > self.max_speed:
            v = v / np.linalg.norm(v) * self.max_speed

        p = np.clip(p + v * self.dt, -self.max_position, self.max_position)

        distance = np.linalg.norm(g-p)
        reward += np.dot(v, g-p) / distance - .001*(np.linalg.norm(u)**2)
        # reward += -1 * (distance / 40) ** 2

        done = distance < self.treshold
        if done:
            reward += self.goal_reward

        self.state = np.concatenate([p, g, v])
        return self._get_obs(), float(reward), done, {}

    def reset(self):
        high = self.observation_space.high
        self.state = self.np_random.uniform(low=-high, high=high)
        if np.linalg.norm(self.state[-2:]) > self.max_speed:
            self.state[-2:] = self.state[-2:] / np.linalg.norm(self.state[-2:]) * self.max_speed
        if not self.randomize_goal:
            self.state[2:4] = 0
        self.last_u = np.array([0,0])
        # self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
        # theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.max_position, self.max_position, -self.max_position, self.max_position)
            
            point = rendering.make_circle(1)
            point.set_color(.1, .8, .3)
            self.point_transform = rendering.Transform()
            point.add_attr(self.point_transform)
            self.viewer.add_geom(point)

            goal = rendering.make_circle(1)
            goal.set_color(.9, .1, .1)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)
            fname = path.join(path.dirname(__file__), "assets/arrow.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.img_trans = rendering.Transform()
            self.img.add_attr(self.img_trans)

        self.viewer.add_onetime(self.img)
        self.point_transform.set_translation(self.state[0], self.state[1])
        self.goal_transform.set_translation(self.state[2], self.state[3])
        # if self.last_u:
        self.img_trans.set_translation(self.state[0], self.state[1])
        self.img_trans.set_rotation(np.arctan2(self.last_u[1], self.last_u[0]))
        scale = np.linalg.norm(self.last_u) / self.max_torque * 2
        self.img_trans.set_scale(scale, scale)
        # import time
        # time.sleep(self.dt)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def visualize_solution(self, policy=None, value_func=None, i_episode=None):
        '''
            @brief Visualizes policy and value functions
            the policy/value are visualized only for states where goal = v = 0

            @param policy: a function that takes the state as input and outputs the action (as numpy array)
            @param valud_func: a function that takes the state as input and outputs the value (as float)
        '''
        nb_points = 20
        xlim = [self.observation_space.low[0], self.observation_space.high[0]]
        ylim = [self.observation_space.low[1], self.observation_space.high[1]]

        if self.plot is None:
            self.plot = Plot(1,2)
            self.splot = ScatterPlot(
                parent=self.plot,
                xlim=xlim, ylim=ylim,
                value_range=[-self.max_speed*5, self.max_speed*5]
            )
            self.qplot = QuiverPlot(parent=self.plot, xlim=xlim, ylim=ylim)
        
        x = np.linspace(xlim[0], xlim[1], nb_points)
        y = np.linspace(ylim[0], ylim[1], nb_points)
        points = np.array(np.meshgrid(x,y)).transpose().reshape((-1,2))
        v = np.ones(points.shape[0])
        d = np.ones((points.shape[0], 2))
        for i, p in enumerate(points):
            state = np.concatenate([p, [0,0,0,0]])
            if value_func is not None:
                v[i] = value_func(state)
            if policy is not None:
                d[i] = policy(state)

        self.splot.update(points, v)
        self.qplot.update(points, d)
        
        if i_episode is not None and self.writer is not None:
            self.writer.add_image('Vis/Nets', self.plot.get_image(), i_episode)

    def close(self):
        if self.viewer:
            self.viewer.close()
