
from gym.utils import seeding
import numpy as np
import os.path as path
import copy
import gym
import pybullet_envs

from algorithms.plot import ScatterPlot, QuiverPlot, Plot
from core.object_utils import ObjectWrapper
from algorithms.normalization import NormalizedEnv

class PointMass(gym.Env):
    '''
    Just a simple 2D PointMass with a jet, trying to go towards a goal location
    '''
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }
    reward_styles = {
        'velocity': dict(vel=1, pos=0, goal=100),
        'distsq'  : dict(vel=0, pos=1, goal=0),
        'distsq+g': dict(vel=0, pos=1, goal=5),
    }

    def __init__(self, max_steps=100, randomize_goal=True, writer=None, reset=True, reward_style='velocity'):
        self.max_speed = 2.
        self.max_torque = 2.
        self.max_position = 20.
        self.max_steps = max_steps
        self.treshold = 2.
        self.dt = .1
        self.mass = .2
        self.randomize_goal = randomize_goal
        
        if reward_style in self.reward_styles:
            self.reward_style = self.reward_styles[reward_style]
        else:
            raise ValueError(
                'Incorrent `reward_style` argument %s. Should be one of [%s]'
                % (reward_style, ', '.join(self.reward_styles.keys()))
            )

        self.writer = writer

        self.viewer = None
        self.plot = None

        # self.obs_size = 4
        self.act_size = 2
        
        high_action = self.max_torque * np.ones(self.act_size)
        self.action_space = gym.spaces.Box(low=-high_action, high=high_action, dtype=np.float32)

        self.obs_high = np.concatenate([self.max_position * np.ones(4), self.max_speed * np.ones(2)])
        self.observation_space = gym.spaces.Box(low=-self.obs_high, high=self.obs_high, dtype=np.float32)

        self.seed()
        if reset:
            self.reset()
    
    # @property
    # def observation_space(self):
    #     high_position = np.concatenate([self.max_position * np.ones(4), self.max_speed * np.ones(2)])
    #     return gym.spaces.Box(low=-high_position, high=high_position, dtype=np.float32)

    # @property
    # def action_space(self):
    #     high_action = self.max_torque * np.ones(self.act_size)
    #     return gym.spaces.Box(low=-high_action, high=high_action, dtype=np.float32)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def step(self, u):
        self.i_step += 1

        # state: (px, py, gx, gy, vx, vy)
        p = self.state[0:2]
        g = self.state[2:4]
        v = self.state[4:6]
        
        reward = 0

        u = np.array(u)  # PyTorch poblems: fixing it from both sides

        if np.linalg.norm(u) > self.max_torque:
            u = u / np.linalg.norm(u) * self.max_torque
        
        self.last_u = u

        p, v = self._integrate(p, v, u)

        distance = np.linalg.norm(g-p)
        reward += self.reward_style['vel'] * np.dot(v, g-p) / distance - .001*(np.linalg.norm(u)**2)
        reward -= self.reward_style['pos'] * (distance / self.max_position) ** 2

        reached = distance < self.treshold
        if reached:
            reward += self.reward_style['goal']
        
        done = reached or (self.i_step >= self.max_steps)

        self.state = np.concatenate([p, g, v])
        return self._get_obs(), float(reward), done, {}

    def _integrate(self, p, v, u):
        # just a simple (dumb) explicit integration ... 
        v = v + u * self.dt / self.mass
        if np.linalg.norm(v) > self.max_speed:
            v = v / np.linalg.norm(v) * self.max_speed

        p = np.clip(p + v * self.dt, -self.max_position, self.max_position)

        return p, v

    def reset(self):
        high = self.obs_high
        self.state = self.np_random.uniform(low=-high, high=high)
        if np.linalg.norm(self.state[-2:]) > self.max_speed:
            self.state[-2:] = self.state[-2:] / np.linalg.norm(self.state[-2:]) * self.max_speed
        if not self.randomize_goal:
            self.state[2:4] = 0
        self.last_u = np.array([0,0])
        self.i_step = 0
        # self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state
        # theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
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

    def visualize_solution(self, policy=None, value_func=None, i_iter=None):
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
            state = np.concatenate([p, [0] * (self.observation_space.shape[0] - 2)])
            if value_func is not None:
                v[i] = value_func(state)
            if policy is not None:
                d[i] = policy(state)

        self.splot.update(points, v)
        self.qplot.update(points, d)
        
        if i_iter is not None and self.writer is not None:
            self.writer.add_image('Vis/Nets', self.plot.get_image(), i_iter)

    def close(self):
        if self.viewer:
            self.viewer.close()


class PointMassV2(PointMass):
    def _integrate(self, cur_p, cur_v, u):
        _, v = super()._integrate(cur_p, cur_v, u)

        p = np.clip(
            cur_p  +  v * self.dt  +  0.5 * u * self.dt * self.dt,
            -self.max_position, self.max_position
        )
        
        return p, v


class NStepPointMass(PointMass):
    def __init__(self, step_multiplier=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_multiplier = step_multiplier

    def step(self, u):
        total_reward = 0
        for i in range(self.step_multiplier):
            state, reward, done, extra = super().step(u)
            total_reward += reward
            if done:
                return state, total_reward / (i+1), done, extra
        return state, total_reward / self.step_multiplier, done, extra


class CircularPointMass(PointMass):
    def __init__(self, radius=None, angular_speed=0.02, start_at_goal=False, *args, **kwargs):
        super().__init__(reset=False, *args, **kwargs)
        self.angular_speed = angular_speed
        self.start_at_goal = start_at_goal
        self.treshold = -1.  # should never achieve the goal, just needs to follow it
        self.radius = radius if radius is not None else self.max_position / 2
        self.reset()

    def reset(self):
        super().reset()
        self.phase = 0
        if self.randomize_goal:
            self.phase = self.np_random.uniform(-np.pi, np.pi)
        self._set_goal_pos()
        if self.start_at_goal:
            self.state[0:2] = self.state[2:4]
        return self._get_obs()
        
    def _set_goal_pos(self):
        self.state[2:4] = np.array([np.cos(self.phase), np.sin(self.phase)]) * self.radius

    def step(self, u):
        self.phase += self.angular_speed
        self._set_goal_pos()
        return super().step(u)


class CircularPointMassSAG(CircularPointMass):
    def __init__(self, *args, **kwargs):
        super().__init__(start_at_goal=True, *args, **kwargs)

class CircularPhaseSAG(CircularPointMass):
    def __init__(self, *args, **kwargs):
        self.phase = 0
        self.parent_indices = [0,1,4,5]
        super().__init__(start_at_goal=True, *args, **kwargs)
        high = np.concatenate([self.observation_space.high[self.parent_indices], [np.pi]])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def _get_obs(self):
        while self.phase > np.pi:
            self.phase -= 2*np.pi
        while self.phase < -np.pi:
            self.phase += 2*np.pi
        return np.concatenate([self.state[self.parent_indices], [self.phase]])

class MultiStepEnv(ObjectWrapper):
    def __init__(self, env, nb_steps=5):
        super().__init__(env)
        self.__wrapped__ = env
        self.nb_steps = nb_steps
    
    def step(self, action):
        total_reward = 0
        for i in range(self.nb_steps):
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward / self.nb_steps, done, {}
    

_ENV_MAP = dict(
    PointMass=PointMass, PM=PointMass,
    PM2=PointMassV2,
    NSPM=NStepPointMass,
    CircularPointMass=CircularPointMass, CPM=CircularPointMass,
    CircularPointSAG=CircularPointMassSAG, CPSAG=CircularPointMassSAG,
    CircularPhaseSAG=CircularPhaseSAG, CPhase=CircularPhaseSAG,
)


# class SerializableEnv(ObjectWrapper):
#     def __init__(self, **kwargs):
#         super().__init__(None)
#         self.__params = kwargs
#         self.__setstate__(kwargs)

#     def __getstate__(self):
#         state = copy.copy(self.__params)
#         # FIXME: not saving this for now, but may change that later
#         state['writer'] = None
#         return state
    
#     def __setstate__(self, state):
#         self.__params = state
#         env = SerializableEnv._get_env(**self.__params)
#         super().set_wrapped(env)

    # @staticmethod
def get_env(name, multi_step=None, **kwargs):
        if name in _ENV_MAP:
            env = _ENV_MAP[name](**kwargs)
        else:
            env = gym.make(name)
        if multi_step:
            env = MultiStepEnv(env, multi_step)
        return env
