import unittest

import numpy as np

from dezero.common import Variable
from dezero.diff import numerical_diff
from dezero.functions import add, square, Square


class TestAdd(unittest.TestCase):

    def test_forward_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        y.backward()
        self.assertEqual(np.array(5.0), y.data)
        self.assertEqual(np.array(1.0), x0.grad)
        self.assertEqual(np.array(1.0), x1.grad)

    def test_using_same_inputs(self):
        x0 = Variable(np.array(2.0))
        y = add(x0, x0)
        y.backward()
        self.assertEqual(np.array(4.0), y.data)
        self.assertEqual(np.array(2.0), x0.grad)

    def test_clear_gradients(self):
        x0 = Variable(np.array(2.0))
        y = add(x0, x0)
        y.backward()
        self.assertEqual(np.array(4.0), y.data)
        self.assertEqual(np.array(2.0), x0.grad)

        x0.clear_grad()
        y = add(add(x0, x0), x0)
        y.backward()
        self.assertEqual(np.array(6.0), y.data)
        self.assertEqual(np.array(3.0), x0.grad)


class TestSquare(unittest.TestCase):

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


class TestTopologicalOrder(unittest.TestCase):

    def test_forward_backward(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertTrue(np.array(32.0), y.data)
        self.assertEqual(np.array(64.0), x.grad)
