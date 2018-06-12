from gym.envs.classic_control import rendering
from gym.utils import seeding
import numpy as np
import gym

class PointMass(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }

    def __init__(self, randomize_goal=True):
        self.max_speed = 2.
        self.max_torque = .5
        self.max_position = 20.
        self.treshold = 1.
        self.dt = .05
        self.viewer = None
        self.randomize_goal = randomize_goal

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

        u = np.clip(u, -self.max_torque, self.max_torque)

        # just a simple (dumb) explicit integration ... 
        v = v + u
        if np.linalg.norm(v) > self.max_speed:
            v = v / np.linalg.norm(v) * self.max_speed

        p = np.clip(p + v * self.dt, -self.max_position, self.max_position)

        distance = np.linalg.norm(g-p)
        reward = np.dot(v, g-p) / distance - .001*(np.linalg.norm(u)**2)

        done = distance < self.treshold
        if done:
            reward += 100

        self.state = np.concatenate([p, g, v])
        return self._get_obs(), float(reward), done, {}

    def reset(self):
        high = self.observation_space.high
        self.state = self.np_random.uniform(low=-high, high=high)
        if np.linalg.norm(self.state[-2:]) > self.max_speed:
            self.state[-2:] = self.state[-2:] / np.linalg.norm(self.state[-2:]) * self.max_speed
        if not self.randomize_goal:
            self.state[2:4] = 0
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

        self.point_transform.set_translation(self.state[0], self.state[1])
        self.goal_transform.set_translation(self.state[2], self.state[3])
        # import time
        # time.sleep(self.dt)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
