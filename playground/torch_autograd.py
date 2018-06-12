'''
    Just exploring the autograd package in PyTorch
    I expected that it would not work when you do multiple forward passes before backward.

    Result: PyTorch seems to be storing ALL the intermediate computations.

    This is pretty handy since it takes care of everything, but you should
    be a little careful when using it: https://pytorch.org/docs/master/notes/faq.html

    examples discussed:
      - accumulating history during training loop
      - holding on to tensors that are not needed
      - using RNNs on long sequences
'''
import torch as th
from nets import *

f = make_net([1,2,2], [th.nn.Tanh()])

f.zero_grad()
y = f(th.tensor([2.1]))
l1 = (0 - y[0] - y[1]) ** 2
l1.backward()
g1 = [p.grad.clone() for p in f.parameters()]

f.zero_grad()
y = f(th.tensor([3.1]))
l2 = (-2 - y[0] - y[1]) ** 2
l2.backward()
g2 = [p.grad.clone() for p in f.parameters()]


f.zero_grad()
y = f(th.tensor([2.1]))
l1 = (0 - y[0] - y[1]) ** 2
y = f(th.tensor([3.1]))
l2 = (-2 - y[0] - y[1]) ** 2
l = l1+l2
l.backward()
g = [p.grad.clone() for p in f.parameters()]
print(g)
print(g1)
print(g2)

# seems to work!