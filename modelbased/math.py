# -*- coding: utf-8 -*-

# Python2 Compatibility
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

import torch as th
import torch.nn as nn


def log_avg_exp(v):
    v = th.DoubleTensor(v)
    m = v.max()
    return m + v.exp().mean().log()