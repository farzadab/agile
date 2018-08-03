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

register(
    id="FixedWalker2DPDRefEnvDM-v0",
    entry_point="envs.envs:FixedWalker2DPDRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="FixedSlowRunnerPDRefEnvDM-v0",
    entry_point="envs.envs:FixedSlowRunnerPDRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="FixedSlowerRunnerPDRefEnvDM-v0",
    entry_point="envs.envs:FixedSlowerRunnerPDRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="FixedRunnerPDRefEnvDM-v0",
    entry_point="envs.envs:FixedRunnerPDRefEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="TRLRunEnvDM-v0",    # reference motion is flawed (upright torso)
    entry_point="envs.envs:TRLRunBadEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="TRLRunEnvDM-v1",    # reference motion is flawed (upright torso)
    entry_point="envs.envs:TRLRunBadPDEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="TRLRunEnvDM-v2",
    entry_point="envs.envs:TRLRunPDEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="TRLWalkerPDEnvDM-v0",
    entry_point="envs.envs:TRLWalkerPDEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="SymTRLWalkerPDEnvDM-v0",
    entry_point="envs.envs:SymmetricTRLWalkerPDEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)

register(
    id="TRLRunEnvDM-v3",
    entry_point="envs.envs:TRLSlowRunPDEnvDM",
    max_episode_steps=1000,
    reward_threshold=2500.0,
)
