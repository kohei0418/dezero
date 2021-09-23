from _weakref import ReferenceType
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional
import weakref

import numpy as np
from numpy import ndarray

from . import Config


def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    name: Optional[str]
    grad: Optional[ndarray]
    generation: int

    def __init__(self, data: ndarray, name: Optional[str] = None):
        if data is not None and not isinstance(data, ndarray):
            raise TypeError('{} is not supported input type'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            p = 'None'
        else:
            p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = []
        seen = set()

        def add_function(g: Function):
            if g not in seen:
                functions.append(g)
                seen.add(g)
                functions.sort(key=lambda h: h.generation)
        add_function(self.creator)

        while functions:
            f = functions.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_function(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().clear_grad()


class Function(metaclass=ABCMeta):
    inputs: Optional[Tuple[Variable]]
    outputs: Optional[List[ReferenceType[Variable]]]
    generation: int

    def __call__(self, *inputs: Variable):
        self.generation = max([x.generation for x in inputs])

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.train:
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: Variable):
        pass

    @abstractmethod
    def backward(self, *gys: Variable):
        pass
