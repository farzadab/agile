from algorithms.PPO import PPO
from algorithms.senv import get_env
from argparser import Args
from logs import LogMaster


ARGS = Args(
    desc='',

    replay_path='',  # if specified, will not train and only replays the learned policy
    store=True,
    render=False,

    env='PointMass',
    env_reward_style='velocity',
    env_max_steps=100,
    env_randomize_goal=True,

    load_path='',
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

# TODO: add layer sizes to Args

def main():

    logm = LogMaster(ARGS)

    env = get_env(
        name=ARGS.env,
        randomize_goal=ARGS.env_randomize_goal,
        max_steps=ARGS.env_max_steps,
        reward_style=ARGS.env_reward_style,
        writer=logm.get_writer(),
    )

    ppo = PPO(
        env, gamma=ARGS.gamma, running_norm=ARGS.running_norm,
        hidden_layer_size=16, nb_layers=0, nb_critic_layers=2,
        # critic_layers=[16,16], actor_layers=[],
        render=ARGS.render, writer=logm.get_writer(),
    )
    if ARGS.load_path:
        ppo.load_models(ARGS.load_path)
    else:
        ppo.sample_normalization(ARGS.normalization_steps)

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
            nb_iters=ARGS.nb_iters,
            nb_max_steps=ARGS.nb_max_steps,
            nb_updates=ARGS.nb_updates,
            batch_size=ARGS.batch_size)
    finally:
        env.close()


def replay(path):
    from algorithms.PPO import ReplayMemory
    import json
    import torch as th

    ARGS.env_max_steps *= 2
    env = get_env(
        name=ARGS.env,
        randomize_goal=ARGS.env_randomize_goal,
        max_steps=ARGS.env_max_steps,
        reward_style=ARGS.env_reward_style,
        writer=None,
    )
    try:
        ppo = PPO(
            env, gamma=ARGS.gamma, running_norm=ARGS.running_norm,
            hidden_layer_size=16, nb_layers=0, nb_critic_layers=2,
            # critic_layers=[16,16], actor_layers=[],
            render=True, writer=None,
        )
        ppo.load_models(path, critic=False)
        policy = ppo.actor.net.state_dict()
        if len(policy) == 2:  # printing out the policy only if it is a linear one (more complex policies are hard to represent)
            print('Policy:')
            norm_matrix = th.cat(
                (
                    th.cat((th.diag(ppo.norm_state.std), th.zeros((1,ppo.norm_state.std.shape[0])))),
                    th.cat((ppo.norm_state.mean, th.FloatTensor([1]))).reshape(-1,1),
                ),
                1
            )
            actor_matrix = th.cat((policy['0.weight'], policy['0.bias'].reshape(-1,1)), 1)
            print(norm_matrix.mm(actor_matrix.t()))

        # print('\npolicy:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.actor.net.state_dict().items()]), sort_keys=True, indent=4))
        # print('\nvalue func:\n', json.dumps(dict([(k,v.tolist()) for k,v in ppo.critic.state_dict().items()]), sort_keys=True, indent=2))
        ppo.actor.log_std[0] = -20

        mem = ReplayMemory(gamma=ARGS.gamma, gae_lambda=0.9)
        ppo.sample_episode(ARGS.nb_max_steps * 2, mem, 0)
    finally:
        env.close()


if __name__ == '__main__':
    if ARGS.replay_path:
        replay(ARGS.replay_path)
    else:
        main()
