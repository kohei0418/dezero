import unittest

import numpy as np

from dezero import Variable


class TestVariable(unittest.TestCase):

    def test_properties(self):
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

    def test_overloaded_operators(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = np.array(1.0)
        y = (-a / 2.0) ** 3.0 * (b - 0.5) + c
        self.assertEqual(np.array((-3.0 / 2.0) ** 3 * (2.0 - 0.5) + 1.0), y.data)

    def test_with_constants(self):
        x = Variable(np.array(2.0))
        y0 = x + 3.0
        y1 = 3.0 + x
        y2 = np.array([3.0]) + x
        z0 = x * 3.0
        z1 = 3.0 * x
        z2 = np.array([3.0]) * x
        self.assertEqual(np.array(5.0), y0.data)
        self.assertEqual(np.array(5.0), y1.data)
        self.assertEqual(np.array(5.0), y2.data)
        self.assertEqual(np.array(6.0), z0.data)
        self.assertEqual(np.array(6.0), z1.data)
        self.assertEqual(np.array(6.0), z2.data)
