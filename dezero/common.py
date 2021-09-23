import typing
from abc import ABCMeta, abstractmethod

from numpy import ndarray


class Variable:
    grad: typing.Optional[ndarray]

    def __init__(self, data: ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x: Variable = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function(metaclass=ABCMeta):
    input: typing.Optional[Variable]
    output: typing.Optional[Variable]

    def __call__(self, input: Variable):
        self.input = input
        y = self.forward(input.data)
        output = Variable(y)
        output.set_creator(self)
        self.output = output
        return output

    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        pass

    @abstractmethod
    def backward(self, gy: ndarray) -> ndarray:
        pass
