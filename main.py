import torch.nn as nn
import torch
import argparse
import time
import gym
import random
import ipdb

import cust_envs  # Ben's Envs: https://github.com/belinghy/research-experiments

from controllers import RandomController, MPCcontroller
from dynamics import DynamicsModel, Data
from nets import MOENetwork, make_net
import envs
# from envs import PDCrabEnv
    


def collect_data(env, ctrl, nb_total_steps):
    data = Data()

    while nb_total_steps > 0:
        state = env.reset()
        done = False
        # may need to concat x manually here
        while not done:
            action = ctrl.get_action(state)
            new_state, _, done, _ = env.step(action)
            data.add_point(state, action, new_state)
            state = new_state
            nb_total_steps -= 1
    return data


def main():
    nb_total_steps = 1000
    nb_iterations = 40
    hidden_layers = [256]

    env = gym.make('Crab2DCustomEnv-v0') 
    # env = gym.make('PDCrab2DCustomEnv-v0') 
    # env = gym.make('NabiRosCustomEnv-v0')

    ctrl = rand_ctrl = RandomController(env)


    # ipdb.set_trace()

    f_net = make_net(
        [ctrl.nb_inputs() + ctrl.nb_actions()] + hidden_layers + [ctrl.nb_inputs()],
        [nn.ReLU() for _ in hidden_layers],
    )
    print(ctrl.nb_inputs(), ctrl.nb_actions())

    data = collect_data(env, ctrl, nb_total_steps*10)


    # ipdb.set_trace()

    dynamics = DynamicsModel(env, f_net, data.get_all())
    cost_func = lambda x: -x[3].item()  # refers to vx

    # data.calc_normalizations()
    # dynamics.fit(data)

    mpc_ctrl = MPCcontroller(env, dynamics.predict, cost_func, num_simulated_paths=100, horizon=10, num_mpc_steps=10)

    for i in range(nb_iterations):
        print('Iteration', i)
        new_data = collect_data(env, ctrl, nb_total_steps)
        dynamics.fit(*new_data.get_all())
        data.extend(new_data)
        dynamics.fit(*data.sample(sample_size=4*nb_total_steps))
        # dynamics.fit(*data.get_all())
        if random.random() > 0.5:
            ctrl = rand_ctrl
        else:
            ctrl = mpc_ctrl
    
    env = gym.make('Crab2DCustomEnv-v0')
    # env = gym.make('NabiRosCustomEnv-v0')
    ctrl = MPCcontroller(env, dynamics.predict, cost_func, num_simulated_paths=1000, num_mpc_steps=4)

    env.render(mode='human')
    obs = env.reset()

    for _ in range(100):
        # time.sleep(1. / 60.)
        obs, r, done, _ = env.step(ctrl.get_action(obs))
        print('  ', cost_func(obs))
        # if done:
        #     print("done:", r, obs)
            # time.sleep(1)
            # ctrl.reset()
            # obs = env.reset()
    ipdb.set_trace()
    

if __name__ == '__main__':
    main()