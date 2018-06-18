import tensorboardX
import json
import gym
import os

from algorithms.PPO import PPO
from algorithms.senv import PointMass, CircularPointMass
from argparser import Args, ArgsEnum
from jsonfix import dump as json_dump

args = Args(
    nb_iters=400,
    nb_max_steps=1000,
    nb_updates=20,
    batch_size=512,
    normalization_steps=1000,
    running_norm=True,
    gamma=0.9,
    render=False,
    store=True,
    description=ArgsEnum.REQUIRED,
    # TODO: add env here, requires registering my envs ....
    # TODO: add reward function that was used here
).parse()

if __name__ == '__main__':
    import gym
    from algorithms.senv import PointMass
    
    # use logging ....
    writer = log_dir = None
    if args.store:
        writer = tensorboardX.SummaryWriter()
        log_dir = writer.file_writer.get_logdir()
        print('### Logging to `%s`' % log_dir)

    # env = gym.make('Pendulum-v0')
    env = PointMass(randomize_goal=True, writer=writer, max_steps=100)

    # ppo = PPO(env, gamma=0.9, hidden_layers=[4], writer=writer, running_norm=True, render=False)
    ppo = PPO(
        env, gamma=args.gamma, running_norm=args.running_norm,
        critic_layers=[8], actor_layers=[],
        render=args.render, writer=writer,
    )
    ppo.sample_normalization(args.normalization_steps)

    if log_dir:
        variant = dict(
            znet_actor=str(ppo.actor.net),
            znet_critic=str(ppo.critic),
            **vars(args)
        )
        extras = dict(
            norm_rew_mean=str(ppo.norm_rew.mean.tolist()),
            norm_rew_std=str(ppo.norm_rew.std.tolist()),
            norm_state_mean=str(ppo.norm_state.mean.tolist()),
            norm_state_std=str(ppo.norm_state.std.tolist()),
        )
        json_dump(extras, os.path.join(log_dir, 'variant.json'))
        json_dump(extras, os.path.join(log_dir, 'extras.json'))

    try:
        ppo.train(
            nb_iters=args.nb_iters,
            nb_max_steps=args.nb_max_steps,
            nb_updates=args.nb_updates,
            batch_size=args.batch_size)
    finally:
        env.close()