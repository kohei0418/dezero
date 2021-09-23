import contextlib


class Config:
    train = False


@contextlib.contextmanager
def train():
    old_value = getattr(Config, 'train')
    setattr(Config, 'train', True)
    try:
        yield
    finally:
        setattr(Config, 'train', old_value)


is_simple_core = True

if is_simple_core:
    from dezero.core_simple import as_array, as_variable, Variable, Function
    from dezero.core_simple import Square
    from dezero.core_simple import add, sub, rsub, mul, div, rdiv, neg, pow, square, exp


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
