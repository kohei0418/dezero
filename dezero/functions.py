import numpy as np
from numpy import ndarray

from .common import Function, Variable


class Square(Function):

    def forward(self, x: ndarray) -> ndarray:
        return x ** 2

    def backward(self, gy: ndarray) -> ndarray:
        x: ndarray = self.input.data
        return 2 * x * gy


class Exp(Function):

    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)

    def backward(self, gy: ndarray) -> ndarray:
        x: ndarray = self.input.data
        return np.exp(x) * gy


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)
