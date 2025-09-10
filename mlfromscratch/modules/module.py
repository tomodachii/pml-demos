import torch
from torch import nn


class Module:
    """The base class of models."""

    def __init__(self):
        pass

    def loss(self, y_hat, y):
        pass

    def plot(self, key, value, train):
        pass
