import tensorboardX
import logging
import pygit2
import termcolor
import os

from jsonfix import dump as json_dump


logger = logging.getLogger('mine')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


class LogMaster(object):
    def __init__(self, args):
        self.writer = self.log_dir = None
        self.args = args
        if args.store:
            ## prompt the user to provide a desciption
            if (not hasattr(args, 'desc')) or args.desc is None or len(args.desc) == 0:
                print(termcolor.colored(
                    '#### Experiment description required ####\nPlease briefly specify what this experiment is for:',
                    'red'
                ))
                args.desc = input()
            
            args.desc = args.desc

            self.writer = tensorboardX.SummaryWriter()
            self.log_dir = self.writer.file_writer.get_logdir()
            print(termcolor.colored('### Logging to `%s`' % self.log_dir, 'green'))
    
    def get_writer(self):
        return self.writer
    
    def get_logdir(self):
        return self.log_dir
    
    def store_exp_data(self, variant, extras=None):
        if self.log_dir is not None:
            repo = pygit2.Repository('.')
            
            full_variant = dict(
                variant,
                ## added character `z` at the start so that it is printed at the end of the list when sorted
                zgit_commit_id=str(repo.head.target),
                zgit_commit_msg=repo[repo.head.target].message,
                # TODO: command
                **vars(self.args)
            )
            
            json_dump(full_variant, os.path.join(self.log_dir, 'variant.json'))
            
            if extras is not None:
                json_dump(extras, os.path.join(self.log_dir, 'extras.json'))