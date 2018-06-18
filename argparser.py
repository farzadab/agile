'argparse'
import argparse
import enum
import copy

# The option_string is always gonna be --<arg_name>
# And everything is always assumed to be required if the default is not present
_DEFAULT_ARGS = {
    'env': {
        'default': 'PDCrab2DCustomEnv-v0',
        'help': 'Name of the Gym environment to run',
    },
    'mode': {
        'default': 'human',
        'help': 'The render mode for the environment. "human" will open a GUI',
    },
    'description': {
        'required': True,
        'help': 'Description of what the experiment is trying to accomplish',
    },
    'timesteps': {
        'type': int,
        'default': 1000,
        'help': 'For how many timesteps should we run the application',
    },
    'nb_iters': {
        'type': int,
        'help': 'Number of iterations to run the algorithm',
    },
    'batch_size': {
        'type': int,
        'default': 512,
        'help': 'The size of the batch at each optimization step',
    },
    'running_norm': {
        'type': bool,
        'help': 'Whether or not to use a running average for normalization',
    },
    'store': {
        'type': bool,
        'help': 'Whether or not to store the result and logs (tensorboard, variants, etc)',
    },
    'ctrl': {
        'help': 'Name of the controller to use',
    },
}


# TODO: legacy, should remove later
def parse_args(description='', vars_list=[]):
    parser = argparse.ArgumentParser(
        description=description
    )
    for var in vars_list:
        # set option_string to --<arg_name>
        parser.add_argument('--'+var, **_DEFAULT_ARGS[var])
    return parser.parse_args()


class ArgsEnum(enum.Enum):
    REQUIRED = 1


class Args(object):
    def __init__(self, description='', **arguments):
        self.args = dict()

        for key, default in arguments.items():
            details = dict()
            if key in _DEFAULT_ARGS:
                details = copy.deepcopy(_DEFAULT_ARGS[key])

            details['default'] = default

            if default is ArgsEnum.REQUIRED:
                del details['default']
                details['required'] = True
    
            self.args[key] = details

    def parse(self, description=''):
        parser = argparse.ArgumentParser(
            description=description
        )
        for var, details in self.args.items():
            # set option_string to --<arg_name>
            parser.add_argument('--' + var, **details)
        return parser.parse_args()
        