import unittest

import numpy as np

from dezero.common import Variable


class TestProperties(unittest.TestCase):

    def test(self):
        x = Variable(np.eye(3), name='hoge')
        self.assertEqual('hoge', x.name)
        self.assertEqual((3, 3), x.shape)
        self.assertEqual(2, x.ndim)
        self.assertEqual(9, x.size)
        self.assertEqual(np.float64, x.dtype)
        self.assertEqual(3, len(x))
        self.assertEqual('''variable([[1. 0. 0.]
          [0. 1. 0.]
          [0. 0. 1.]])''', repr(x))
