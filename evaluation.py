import torch as th
import numpy as np

from logs import logger

class EvaluationArgs:
    def __init__(self, nb_episodes=10, nb_burnin_steps=10, horizons=[1, 2, 4, 8, 16, 32]):
        self.nb_episodes = nb_episodes
        self.nb_burnin_steps = nb_burnin_steps
        self.horizons = horizons


def evaluate_and_log_dynamics(dynamics, env, ctrl, writer, i_step, args):
    results = []
    for _ in range(args.nb_episodes):
        results.append(_evaluate_dynamics_one_episode(dynamics, env, ctrl, args))
    # print(results)
    aggr = np.nanmean(results, 0).tolist()
    print(aggr)
    for v, h in zip(aggr, args.horizons):
        print(v, h)
        writer.add_scalar('Eval/Horiz%d/%s' % (h, ctrl.get_name()) , v, i_step)


import ipdb

def simulate(dynamics, state, actions):
    predictions = []
    for a in actions:
        predictions.append(dynamics(state, a))
    return list(reversed(predictions))

def compute_distance(obs1, obs2):
    # ipdb.set_trace()
    return np.mean(np.square(np.subtract(obs1, obs2)))


def _evaluate_dynamics_one_episode(dynamics, env, ctrl, eval_args, nb_tries=3):
    obs = env.reset()
    ctrl.reset()
    # ipdb.set_trace()
    for _ in range(eval_args.nb_burnin_steps):
        obs, _, done, _ = env.step(ctrl.get_action(obs))
        if done:
            if nb_tries > 0:
                logger.warn('could not step past the burnin without the episode ending')
                return _evaluate_dynamics_one_episode(dynamics, env, ctrl, eval_args, nb_tries-1)
            else:
                return [np.nan for _ in eval_args.horizons]
    
    nb_steps = max(eval_args.horizons)
    distances = [np.nan for _ in eval_args.horizons]
    # TODO: costs
    actions, extra = ctrl.get_multistep_actions(obs, nb_steps)
    if 'predictions' in extra:
        predictions = extra['predictions']
    else:
        predictions = simulate(dynamics, obs, actions)
    
    # ipdb.set_trace()
    for i in range(nb_steps):
        obs, _, done, _ = env.step(actions[i])
        if i+1 in eval_args.horizons:
            distances[eval_args.horizons.index(i+1)] = compute_distance(obs, predictions[i])
    return distances

