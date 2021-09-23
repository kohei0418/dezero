import unittest

import numpy as np

from dezero.common import Variable
from dezero.diff import numerical_diff
from dezero.functions import square, Square


class SquareTest(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(expected, y.data)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(expected, x.grad)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(Square(), x)
        self.assertTrue(np.allclose(num_grad, x.grad))
