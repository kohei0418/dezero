import weakref
from _weakref import ReferenceType
from abc import ABCMeta, abstractmethod
from typing import Optional, List

import numpy as np
from numpy import ndarray

from dezero import Config


def as_array(x) -> ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    __array_priority__ = 200

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

        def add_function(g):
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


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function(metaclass=ABCMeta):
    inputs: Optional[List[Variable]]
    outputs: Optional[List[ReferenceType[Variable]]]
    generation: Optional[int]

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.train:
            self.generation = max([x.generation for x in inputs])
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
