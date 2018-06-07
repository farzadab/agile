'argparse'
import argparse

# the option_string is always gonna be --<arg_name>
args_list = {
    'env': {
        'default': 'PDCrab2DCustomEnv-v0',
        'help': 'Name of the Gym environment to run',
    },
    'mode': {
        'default': 'human',
        'help': 'The render mode for the environment. "human" will open a GUI',
    },
    'timesteps': {
        'type': int,
        'default': 1000,
        'help': 'For how many timesteps should we run the application',
    },
    'ctrl': {
        'help': 'Name of the controller to use',
    },
}

def parse_args(description='', vars_list=[]):
    parser = argparse.ArgumentParser(
        description=description
    )
    for var in vars_list:
        # set option_string to --<arg_name>
        parser.add_argument('--'+var, **args_list[var])
    return parser.parse_args()