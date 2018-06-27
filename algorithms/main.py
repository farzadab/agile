from algorithms.PPO import PPO
from algorithms.senv import get_env
from argparser import Args
from logs import LogMaster, ConsoleWriter, AverageWriter

def get_args():
    return Args(
        desc='',

        replay_path='',  # if specified, will not train and only replays the learned policy
        store=True,
        render=False,

        env='PointMass',
        env_reward_style='velocity',
        env_max_steps=100,
        env_randomize_goal=True,

        net_layer_size=16,
        net_nb_layers=0,
        net_nb_critic_layers=2,

        load_path='',
        gamma=0.9,
        gae_lambda=0.8,
        noise=-1,
        nb_iters=400,
        nb_max_steps=1000,
        nb_updates=20,
        batch_size=512,
        normalization_steps=1000,
        running_norm=True,
    ).parse()


def main():
    args = get_args()

    writer = None
    if args.replay_path:
        args.store = False

    logm = LogMaster(args)
    writer = logm.get_writer()

    env = get_env(
        name=args.env,
        randomize_goal=args.env_randomize_goal,
        max_steps=args.env_max_steps,
        reward_style=args.env_reward_style,
        writer=writer,
    )

    ppo = PPO(
        env, gamma=args.gamma, gae_lambda=args.gae_lambda,
        running_norm=args.running_norm,
        exploration_noise=args.noise,
        hidden_layer_size=args.net_layer_size,
        nb_layers=args.net_nb_layers, nb_critic_layers=args.net_nb_critic_layers,
        writer=writer,
        render=args.render or bool(args.replay_path),  # always render in replay mode
    )

    if args.replay_path:
        replay(args, env, ppo)
    else:
        train(args, env, ppo, logm)

    if writer is not None:
        writer.flush()


def train(args, env, ppo, logm=None):

    if args.load_path:
        ppo.load_models(args.load_path)
    else:
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


def replay(args, env, ppo):
    from algorithms.PPO import ReplayMemory
    import json
    import torch as th

    # args.env_max_steps *= 2
    try:
        ppo.load_models(args.replay_path, critic=False)
        if ppo.actor.is_linear():
            print('Policy:')
            print(ppo.extract_linear_policy())

        # print('\npolicy:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.actor.net.state_dict().items()]), sort_keys=True, indent=4))
        # print('\nvalue func:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.critic.state_dict().items()]), sort_keys=True, indent=2))

        ppo.actor.log_std[0] = -20  # simple hack to decrease the exploration level

        mem = ReplayMemory(gamma=args.gamma, gae_lambda=0.9)
        ppo.sample_episode(args.nb_max_steps, mem, 0)
    finally:
        env.close()


if __name__ == '__main__':
    main()

