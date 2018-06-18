# Just a simple environment tester, based on https://github.com/FracturedPlane/RLSimulationEnvironments/blob/master/EnvTester.py
import numpy as np
import argparse
import ipdb
import time
import gym

# load my environments
import cust_envs, envs
from algorithms.senv import get_env

def parse_args():
    parser = argparse.ArgumentParser(
        description="Just running an environment"
    )
    parser.add_argument(
        "--env", default="Crab2DCustomEnv-v0", help="Name of Gym environment to run"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print('Testing environment %s' % args.env)
    env = get_env(args.env)
    print(env)
    
    try:
        env.render(mode='human')
    except:
        env.reset()
        env.render(mode='human')

    actionSpace = env.action_space
    # env.setRandomSeed(1234)
    
    print("observation_space: ", env.observation_space.low)
    print("observation_space: ", env.observation_space.high)
    print("Actions space max: ", len(env.action_space.high))
    print("Actions space min: ", env.action_space.low)
    print("Actions space max: ", env.action_space.high)
    
    env.reset()
    
    for epoch in range(10):
        env.reset()
        print ("New episode")
        for state in range(100):
            env.render(mode='human')
            # actions = []
            # for i in range(env.getNumberofAgents()):
            #     action = ((actionSpace.high - actionSpace.low) * np.random.uniform(size=actionSpace.low.shape[0])  ) + actionSpace.low
            #     actions.append(action)
            # if (env.getNumberofAgents() > 1):
            #     observation, reward,  done, info = env.step(actions)
            # else:
            time.sleep(1 / 60)
            action = actionSpace.sample()
            observation, reward,  done, info = env.step(actionSpace.sample())
            # observation, reward,  done, info = env.step(np.ones(actionSpace.shape))
            print ("Reward: ", reward, "Action: ", action, " observation: ", observation)
            print ("Done: ", done)
            if env.action_space.shape != action.shape:
                raise ValueError('Action space and action do not match!')
            if env.observation_space.shape != observation.shape:
                raise ValueError('Observation space and observation do not match!')
            if ( done ):
                break

    ipdb.set_trace()   
            

if __name__ == '__main__':
    main()