import numpy as np
from numpy import ndarray

from .common import as_array, Function, Variable


class Add(Function):

    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


class Sub(Function):

    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


class Mul(Function):

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):

    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy / x1, gy * (-x0 / x1 ** 2)


class Neg(Function):

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Pow(Function):

    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0].data
        return self.c * x ** (self.c - 1) * gy


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
    return Add()(x0, as_array(x1))


def sub(x0, x1):
    return Sub()(x0, as_array(x1))


def rsub(x0, x1):
    return Sub()(as_array(x1), x0)


def mul(x0, x1):
    return Mul()(x0, as_array(x1))


def div(x0, x1):
    return Div()(x0, as_array(x1))


def rdiv(x0, x1):
    return Div()(as_array(x1), x0)


def neg(x):
    return Neg()(x)


def pow(x, c):
    return Pow(c)(x)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__neg__ = neg
Variable.__pow__ = pow
