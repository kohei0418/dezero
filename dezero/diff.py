from numpy import ndarray

from .core_simple import Variable, Function


def numerical_diff(f: Function, x: Variable, eps: float=1e-4) -> ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
