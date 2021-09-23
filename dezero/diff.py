from numpy import ndarray

from common import Function, Variable


def numerical_diff(f: Function, x: Variable, eps: float=1e-4) -> ndarray:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == '__main__':
    from functions import Exp, Square
    import numpy as np
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input == b

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
