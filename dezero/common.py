from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from numpy import ndarray


def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    grad: Optional[ndarray]

    def __init__(self, data: ndarray):
        if data is not None and not isinstance(data, ndarray):
            raise TypeError('{} is not supported input type'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = [self.creator]
        while functions:
            f = functions.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    functions.append(x.creator)


class Function(metaclass=ABCMeta):
    inputs: Optional[Tuple[Variable]]
    outputs: Optional[List[Variable]]

    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: Variable):
        pass

    @abstractmethod
    def backward(self, *gys: Variable):
        pass
