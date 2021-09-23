import numpy as np
from numpy import ndarray

from .common import Function, Variable


class Add(Function):

    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x: ndarray = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x: ndarray = self.inputs[0].data
        return np.exp(x) * gy


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)
