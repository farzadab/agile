'''Run manual controllers on an environment'''
import torch.nn as nn
import torch
import time
import gym
import random
import ipdb

import cust_envs  # Ben's Envs: https://github.com/belinghy/pybullet-custom-envs

from manualControllers import leanLeftRightController
from argparser import parse_args
import envs
from costs import get_cost
# from envs import PDCrabEnv
    


def main():
    args = parse_args(__doc__, ['env', 'mode', 'timesteps'])
    
    env = gym.make(args.env)
    env.render(mode=args.mode)
    obs = env.reset()
    env.unwrapped.set_gravity(-1)

    # TODO: use controller arg
    ctrl = leanLeftRightController(env)

    for _ in range(args.timesteps):
        obs, _, done, _ = env.step(ctrl.get_action(obs))
        if done:
            print('\n=============\nThe episode has finished, restarting\n=============\n')
            time.sleep(1)
            obs = env.reset()
            ctrl.reset()





if __name__ == '__main__':
    main()