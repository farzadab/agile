from algorithms.PPO import PPO
from algorithms.senv import get_env
from argparser import Args
from logs import LogMaster


def main():
    args = Args(
        desc='',

        store=True,
        render=False,

        env_name='PointMass',
        env_reward_style='velocity',
        env_max_steps=100,
        env_randomize_goal=True,

        gamma=0.9,
        nb_iters=400,
        nb_max_steps=1000,
        nb_updates=20,
        batch_size=512,
        normalization_steps=1000,
        running_norm=True,
        # TODO: add env here, requires registering my envs ....
        # TODO: add reward function that was used here
    ).parse()

    logm = LogMaster(args)

    env = get_env(
        name=args.env_name,
        randomize_goal=args.env_randomize_goal,
        max_steps=args.env_max_steps,
        reward_style=args.env_reward_style,
        writer=logm.get_writer(),
    )

    ppo = PPO(
        env, gamma=args.gamma, running_norm=args.running_norm,
        critic_layers=[8], actor_layers=[],
        render=args.render, writer=logm.get_writer(),
    )
    ppo.sample_normalization(args.normalization_steps)

    logm.store_exp_data(
        variant=dict(  # automatically stores args, commit ID, etc
            znet_actor=str(ppo.actor.net),
            znet_critic=str(ppo.critic),
        ),
        extras=dict(
            norm_rew_mean=str(ppo.norm_rew.mean.tolist()),
            norm_rew_std=str(ppo.norm_rew.std.tolist()),
            norm_state_mean=str(ppo.norm_state.mean.tolist()),
            norm_state_std=str(ppo.norm_state.std.tolist()),
        )
    )

    try:
        ppo.train(
            nb_iters=args.nb_iters,
            nb_max_steps=args.nb_max_steps,
            nb_updates=args.nb_updates,
            batch_size=args.batch_size)
    finally:
        env.close()


if __name__ == '__main__':
    main()
