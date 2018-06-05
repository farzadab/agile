from envs import HalfCheetahEnvNew

def crab_cost(state, action, next_state):
    return -next_state[3].item()  # refers to vx


env_cost_map = {
    'Crab2DCustomEnv-v0': crab_cost,
    'HalfCheetahNew-v0': HalfCheetahEnvNew.calc_reward,

}

def get_cost(env_name):
    if env_name in env_cost_map:
        return env_cost_map[env_name]
    else:
        raise ValueError('Cost function for environment %s not found' % str(env_name))