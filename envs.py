import numpy as np
import gym
import cust_envs
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
from gym.envs.registration import registry, make, spec


class HalfCheetahEnvNew(HalfCheetahBulletEnv):
    # def __init__(self):
    #     super().__init__()
    #     self.ordered_joints_names = [j.joint_name for j in self.robot.ordered_joints]
    #     # not really correct, since x can go on
    # #     mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
    # #     utils.EzPickle.__init__(self)

    # def _step(self, action):
    #     obs, reward, done, extra = super()._step(action)
    #     return np.concatenate([obs, self.robot.robot_body.pose().xyz()]), reward, done, extra
        # xposbefore = self.model.data.qpos[0, 0]
        # self.do_simulation(action, self.frame_skip)
        # xposafter = self.model.data.qpos[0, 0]
        # ob = self._get_obs()
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        # reward = reward_ctrl + reward_run
        # done = False
        # return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
    
    # def _reset(self):
    #     obs = super()._reset()
    #     return np.concatenate([obs, self.robot.robot_body.pose().xyz()])

    @staticmethod
    def get_joint_state(joint_name, state):
        '''
            helper function to better interpret the state variables
        '''
        ordered_joints_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        index_start = 8 + 2 * ordered_joints_names.index(joint_name)
        return state[index_start:index_start+1]

    @staticmethod
    def calc_reward(state, action, next_state):
        '''
            calculate the reward just based on (s, a, ns) pair
        '''
        reward = -10 * next_state[3]  # refers to vx
        reward += 0.05 * np.sum(np.square(action))  # penalizing high energy consumption

        for joint in ['fthigh', 'fshin', 'ffoot']:
            joint_pos = HalfCheetahEnvNew.get_joint_state(joint, next_state)[0]
            if joint_pos > 0.2 :
                reward += 1  # penalty
        
        return reward


    

    # def _get_obs(self):
    #     body_pose = self.robot_body.pose()
    #     return np.concatenate([
    #         self.model.data.qpos.flat[1:],
    #         self.model.data.qvel.flat,
    #         self.get_body_com("torso").flat,
    #         # self.get_body_comvel("torso").flat,
    #     ])

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

# class PDController:

#     def __init__(self, env):
#         self.action_dim = env.action_space.shape[0]
#         self.high = env.action_space.high
#         self.low = env.action_space.low
#         frequency = 2
#         self.k_p = (2 * np.pi * frequency) ** 2
#         damping_ratio = 1
#         self.k_d = 2 * damping_ratio * 2 * np.pi * frequency

#     def drive_torques(self, targets, states):
#         # States and targets should be [theta, omega] * action_dim
#         diff = targets - states
#         torques = self.k_p * diff[0::2] + self.k_d * diff[1::2]
#         return np.clip(torques, self.low, self.high)

       
def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

register(
    id="HalfCheetahNew-v0",
    entry_point="envs:HalfCheetahEnvNew",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)
