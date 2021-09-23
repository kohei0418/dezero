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
