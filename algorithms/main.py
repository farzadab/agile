from algorithms.PPO import PPO
from algorithms.senv import SerializableEnv
from argparser import Args
from logs import LogMaster, ConsoleWriter, AverageWriter
from algorithms.anneal import LinearAnneal
from algorithms.normalization import NormalizedEnv

def get_args():
    return Args(
        desc='',

        replay_path='',  # if specified, will not train and only replays the learned policy
        store=True,
        render=False,
        load_path='',

        env='PointMass',
        env_reward_style='velocity',
        env_max_steps=100,
        env_randomize_goal=True,

        multi_step=None,

        net_layer_size=16,
        net_nb_layers=0,
        net_nb_critic_layers=2,

        step_size=0.0003,

        gamma=0.99,
        gae_lambda=0.95,
        noise=-1,  # TODO: anneal noise: LinearAnneal(-0.7, -1.6)
        explore_ratio=1.0,

        nb_iters=400,
        nb_epochs=10,
        mini_batch_size=512,
        batch_size=512*8,

        normalization_steps=1000,
        running_norm=True,
    ).parse()


def main():
    args = get_args()

    writer = None
    if args.replay_path:
        args.store = False
        args.load_path = args.replay_path

    logm = LogMaster(args)
    writer = logm.get_writer()

    if args.load_path:
        # FIXME: writer!
        print('loading ..... ', args.load_path)
        ppo = PPO.load(args.load_path)
        env = ppo.get_env()
        ppo.writer = writer
        ppo.save_path = writer.get_logdir()
        # env.writer = writer
    else:
        env = NormalizedEnv(
            SerializableEnv(
                name=args.env,
                multi_step=args.multi_step,
                randomize_goal=args.env_randomize_goal,
                max_steps=args.env_max_steps,
                reward_style=args.env_reward_style,
                writer=writer,
            ),
            normalize_obs=True,
            gamma=args.gamma,
        )

        ppo = PPO(
            env, gamma=args.gamma, gae_lambda=args.gae_lambda,
            exploration_noise=args.noise,
            explore_ratio=args.explore_ratio,
            # exploration_anneal=LinearAnneal(-0.7, -1.6),
            init_lr=args.step_size,
            hidden_layer_size=args.net_layer_size,
            nb_layers=args.net_nb_layers, nb_critic_layers=args.net_nb_critic_layers,
            writer=writer,
            render=args.render or bool(args.replay_path),  # always render in replay mode
        )

    if args.replay_path:
        try:
            env.render(mode='human')
        except Exception as err:
            print('Exception:', err)
        replay(args, env, ppo)
    else:
        train(args, env, ppo, logm)

    if writer is not None:
        writer.flush()


def train(args, env, ppo, logm=None):

    if args.load_path:
        # ppo.load_models(args.load_path)
        pass
    else:
        ppo.sample_normalization(args.normalization_steps)

    logm.store_exp_data(
        variant=dict(  # automatically stores args, commit ID, etc
            znet_actor=str(ppo.actor.net),
            znet_critic=str(ppo.critic),
        ),
        extras=dict(
            norm_state_mean=str(env.norm_stt.mean.tolist()),
            norm_state_std=str(env.norm_stt.std.tolist()),
        )
    )

    try:
        ppo.train(
            nb_iters=args.nb_iters,
            batch_size=args.batch_size,
            nb_epochs=args.nb_epochs,
            mini_batch_size=args.mini_batch_size)
    finally:
        env.close()


def replay(args, env, ppo):
    from algorithms.PPO import ReplayMemory

    ppo.render = True

    # args.env_max_steps *= 2
    try:
        # ppo.load_models(args.replay_path, critic=False)
        if ppo.actor.is_linear():
            print('Policy:')
            print(ppo.extract_linear_policy())

        # print('\npolicy:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.actor.net.state_dict().items()]), sort_keys=True, indent=4))
        # print('\nvalue func:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.critic.state_dict().items()]), sort_keys=True, indent=2))

        ppo.actor.log_std[0] = -20  # simple hack to decrease the exploration level

        mem = ReplayMemory(gamma=args.gamma, gae_lambda=0)
        ppo.sample_episode(args.batch_size, mem)
    finally:
        env.close()


if __name__ == '__main__':
    main()
