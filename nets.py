import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import datetime


class NNModel(object):
    def __init__(self, net, optimizer=None, criterion=None, batch_size=1024):
        # datetime.datetime.now().strftime('%Y/%m/%d-%X') + '_net.log'
        self.batch_size = batch_size
        self.writer = SummaryWriter()
        self.net = net.double()
        self.i_iteration = 0
        if optimizer is None:
            # self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)
            self.optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0.0003)
        if criterion is None:
            self.criterion = nn.MSELoss()
    
    def fit_batch(self, X, y):
        self.net.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        yhat = self.net.forward(X)
        loss = self.criterion(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.i_iteration += 1
        
        # write summary to PyTorch-Tensorboard
        self.writer.add_scalar('Train/Loss', loss, self.i_iteration)
        return loss

    def fit(self, X, y):
        X = torch.DoubleTensor(X)
        y = torch.DoubleTensor(y)
        for i in range(0, X.shape[0], self.batch_size):
            self.fit_batch(X[i:i+self.batch_size], y[i:i+self.batch_size])
        # TODO: return loss? have to adjust for the last batch

    def predict(self, x):
        self.net.eval()
        return self.net.forward(x)


class MOENetwork(nn.Module):
    def __init__(self, nb_inputs, nb_experts, gait_layers=[16], expert_layers=[]):
        super(MOENetwork, self).__init__()
        self.nb_experts = nb_experts
        self.experts = []
        self.gaiting_net = make_net(
            [nb_inputs] + gait_layers + [nb_experts],
            [nn.ReLU() for _ in gait_layers] + [nn.Softmax()]
        )
        for _ in nb_experts:
            self.experts.append(make_net(
                [nb_inputs] + expert_layers,
                [nn.ReLU() for _ in expert_layers] + [nn.Tanh()]
            ))
    
    def forward(self, x):
        probs = self.gaiting_net.forward(x)
        res = self.experts[0].forward(x) * probs[0]
        for i in range(1, self.nb_experts):
            res.add_(self.experts[i].forward(x) * probs[i])
        return res


def make_net(dims, activations):
    layers = []
    for i, (inp, out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(inp, out))
        if len(activations) > i:
            layers.append(activations[i])
    return nn.Sequential(*layers)