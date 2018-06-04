import numpy as np
from cust_envs.envs import Crab2DCustomEnv

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

       
# def register(id, *args, **kvargs):
#     if id in registry.env_specs:
#         return
#     else:
#         return gym.envs.registration.register(id, *args, **kvargs)

# register(
#     id="PDCrabCustomEnv-v0",
#     entry_point="envs:PDCrabEnv",
#     max_episode_steps=1000,
#     reward_threshold=2500.0,
# )

# register(
#     id="NabiRosCustomEnv-v0",
#     entry_point="envs:NabiRosEnv",
#     max_episode_steps=1000,
#     reward_threshold=2500.0,
# )