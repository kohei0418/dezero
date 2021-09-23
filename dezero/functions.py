import numpy as np
from numpy import ndarray

from .common import Function, Variable


class Add(Function):

    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


class Mul(Function):

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


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


def add(x0, x1):
    return Add()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


Variable.__mul__ = mul
Variable.__add__ = add
