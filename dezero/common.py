import typing
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray


def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    grad: typing.Optional[ndarray]

    def __init__(self, data: ndarray):
        if data is not None and not isinstance(data, ndarray):
            raise TypeError('{} is not supported input type'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = [self.creator]
        while functions:
            f = functions.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                functions.append(x.creator)


class Function(metaclass=ABCMeta):
    input: typing.Optional[Variable]
    output: typing.Optional[Variable]

    def __call__(self, input: Variable):
        self.input = input
        y = self.forward(input.data)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.output = output
        return output

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def backward(self, gy: ndarray) -> ndarray:
        pass
