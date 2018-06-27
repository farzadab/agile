import tensorboardX
import numpy as np
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
        self.log_dir = None
        self.args = args
        if args.store:
            ## prompt the user to provide a desciption
            if (not hasattr(args, 'desc')) or args.desc is None or len(args.desc) == 0:
                print(termcolor.colored(
                    '#### Experiment description required ####\nPlease briefly specify what this experiment is for:',
                    'red'
                ))
                args.desc = input()
            writer = tensorboardX.SummaryWriter()
        else:
            writer = ConsoleWriter()

        self.writer = AverageWriter(writer)
        self.log_dir = self.writer.get_logdir()

        if self.log_dir is not None:
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


class ConsoleWriter(object):
    def __init__(self):
        self.epoch = -1
        self.file_writer = self  # just so that the command `writer.file_writer.get_logdir()` works
    def add_scalar(self, name, value, epoch):
        if self.epoch != epoch:
            print(
                '============ Iter %d =============' % epoch
            )
        self.epoch = epoch
        print('|{:20s}|{:10.4f}|'.format(name, value))
    def get_logdir(self):
        return None


class AverageWriter(object):
    def __init__(self, writer):
        self.epoch = 0
        self.writer = writer
        self.dict = {}
    def write_values(self):
        for k in sorted(self.dict.keys()):
            self.writer.add_scalar(k, np.mean(self.dict[k]), self.epoch)
        self.dict.clear()
    def set_epoch(self, epoch):
        if self.epoch != epoch:
            self.write_values()
        self.epoch = epoch
    def add_scalar(self, name, value, epoch=None):
        if epoch is not None:
            self.set_epoch(epoch)
        if name not in self.dict:
            self.dict[name] = []
        self.dict[name].append(value)
    # def get_writer(self):
    #     return self.writer.get_writer
    def get_logdir(self):
        return self.writer.file_writer.get_logdir()
    def flush(self):
        self.write_values()

