from envs import HalfCheetahEnvNew

def crab_cost(state, action, next_state):
    cost = -next_state[3].item()   # refers to vx

    if next_state[0].item() < -0.4:
        cost += 10
    if abs(next_state[7].item()) > 1:
        cost += 10
    return cost


env_cost_map = {
    'Crab2DCustomEnv-v0': crab_cost,
    'HalfCheetahNew-v0': HalfCheetahEnvNew.calc_reward,

    'NabiRosCustomEnv-v0': crab_cost,
    'PDCrab2DCustomEnv-v0': crab_cost,
}

def get_cost(env_name):
    if env_name in env_cost_map:
        return env_cost_map[env_name]
    else:
        raise ValueError('Cost function for environment %s not found' % str(env_name))