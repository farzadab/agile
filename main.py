'''MPC with MoE neural net'''
import tensorboardX
import torch.nn as nn
import torch
import time
import gym
import random
import ipdb

import cust_envs  # Ben's Envs: https://github.com/belinghy/pybullet-custom-envs

from controllers import RandomController, MPCcontroller
from dynamics import DynamicsModel, Data
from nets import MOENetwork, make_net
from argparser import parse_args
from evaluation import evaluate_and_log_dynamics, EvaluationArgs
import envs
from costs import get_cost
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
    hidden_layers = [256, 256]
    writer = tensorboardX.SummaryWriter()

    args = parse_args(__doc__, ['env'])


    env = gym.make(args.env) 

    ctrl = rand_ctrl = RandomController(env)


    # ipdb.set_trace()
    print('#inputs : %d' % ctrl.nb_inputs())
    print('#actions: %d' % ctrl.nb_actions())

    # f_net = make_net(
    #     [ctrl.nb_inputs() + ctrl.nb_actions()] + hidden_layers + [ctrl.nb_inputs()],
    #     [nn.ReLU() for _ in hidden_layers],
    # )
    f_net = MOENetwork(
        nb_inputs=ctrl.nb_inputs() + ctrl.nb_actions(),
        nb_experts=4,
        gait_layers=[64],
        expert_layers=[64, ctrl.nb_inputs()],
    )

    data = collect_data(env, ctrl, nb_total_steps*10)


    # ipdb.set_trace()

    dynamics = DynamicsModel(env, f_net, data.get_all(), writer=writer)
    # cost_func = lambda s,a,sn: -sn[3].item()  # refers to vx
    cost_func = get_cost(args.env)  # refers to vx

    # data.calc_normalizations()
    # dynamics.fit(data)

    mpc_ctrl = MPCcontroller(env, dynamics.predict, cost_func, num_simulated_paths=100, horizon=10, num_mpc_steps=10)
    eval_args = EvaluationArgs(nb_burnin_steps=4, nb_episodes=10, horizons=[1, 2, 4, 8, 16, 32])

    for i in range(nb_iterations):
        print('Iteration', i)
        new_data = collect_data(env, ctrl, nb_total_steps)
        dynamics.fit(*new_data.get_all())
        data.extend(new_data)
        dynamics.fit(*data.sample(sample_size=4*nb_total_steps))
        evaluate_and_log_dynamics(
            dynamics.predict, env, rand_ctrl, writer=writer, i_step=i, args=eval_args
        )
        evaluate_and_log_dynamics(
            dynamics.predict, env, mpc_ctrl, writer=writer, i_step=i, args=eval_args
        )
        # dynamics.fit(*data.get_all())
        if random.random() > 0.5:
            ctrl = rand_ctrl
        else:
            ctrl = mpc_ctrl
    
    env = gym.make(args.env)

    ctrl = MPCcontroller(env, dynamics.predict, cost_func, num_simulated_paths=1000, num_mpc_steps=4)

    # TODO

    env.render(mode='human')
    obs = env.reset()

    for _ in range(100):
        # time.sleep(1. / 60.)
        obs, r, done, _ = env.step(ctrl.get_action(obs))
        # print('  ', cost_func(obs))
        if done:
            print("done:", r, obs)
            time.sleep(1)
            ctrl.reset()
            obs = env.reset()
    ipdb.set_trace()
    

if __name__ == '__main__':
    main()