import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

register(
    id="Walker2DEnv-v0",
    entry_point="envs.envs:Walker2DEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="Walker2DRefEnv-v0",
    entry_point="envs.envs:Walker2DRefEnv",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="Walker2DRefEnvDM-v0",
    entry_point="envs.envs:Walker2DRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="Walker2DPDRefEnvDM-v0",
    entry_point="envs.envs:Walker2DPDRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="FastWalker2DRefEnvDM-v0",
    entry_point="envs.envs:FastWalker2DRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="FixedWalkerRefEnvDM-v0",
    entry_point="envs.envs:FixedWalkerRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)
