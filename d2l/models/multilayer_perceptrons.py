import torch
from torch import nn
from utils import *
from .classifier import Classifier


class MLPScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))


class MLP(Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


@add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()


@add_to_class(MLPScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)


@add_to_class(MLPScratch)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)


@add_to_class(MLP)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)


@add_to_class(MLP)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
