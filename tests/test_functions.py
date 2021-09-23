import unittest

import numpy as np

from dezero import add, mul, square, train, Square, Variable
from dezero.diff import numerical_diff


class TestAdd(unittest.TestCase):

    def test_forward_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        with train():
            y = add(x0, x1)
            y.backward()
        self.assertEqual(np.array(5.0), y.data)
        self.assertEqual(np.array(1.0), x0.grad)
        self.assertEqual(np.array(1.0), x1.grad)

    def test_using_same_inputs(self):
        x0 = Variable(np.array(2.0))
        with train():
            y = add(x0, x0)
            y.backward()
        self.assertEqual(np.array(4.0), y.data)
        self.assertEqual(np.array(2.0), x0.grad)

    def test_clear_gradients(self):
        x0 = Variable(np.array(2.0))
        with train():
            y = add(x0, x0)
            y.backward()
        self.assertEqual(np.array(4.0), y.data)
        self.assertEqual(np.array(2.0), x0.grad)

        x0.clear_grad()
        with train():
            y = add(add(x0, x0), x0)
            y.backward()
        self.assertEqual(np.array(6.0), y.data)
        self.assertEqual(np.array(3.0), x0.grad)


class TestMul(unittest.TestCase):

    def test_forward_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        with train():
            y = mul(x0, x1)
            y.backward()
        self.assertEqual(np.array(6.0), y.data)
        self.assertEqual(np.array(3.0), x0.grad)
        self.assertEqual(np.array(2.0), x1.grad)


class TestSquare(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(expected, y.data)

    def test_backward(self):
        x = Variable(np.array(3.0))
        with train():
            y = square(x)
            y.backward()
        expected = np.array(6.0)
        self.assertEqual(expected, x.grad)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        with train():
            y = square(x)
            y.backward()
        num_grad = numerical_diff(Square(), x)
        self.assertTrue(np.allclose(num_grad, x.grad))


class TestTopologicalOrder(unittest.TestCase):

    def test_forward_backward(self):
        x = Variable(np.array(2.0))
        with train():
            a = square(x)
            y = add(square(a), square(a))
            y.backward()
        self.assertEqual(np.array(32.0), y.data)
        self.assertEqual(np.array(64.0), x.grad)

    def test_gradients_of_intermediate_layers(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        with train():
            t = add(x0, x1)
            y = add(x0, t)
            y.backward()
        self.assertEqual(np.array(3.0), y.data)
        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
        self.assertEqual(np.array(2.0), x0.grad)
        self.assertEqual(np.array(1.0), x1.grad)


class TestComplicatedFunctions(unittest.TestCase):

    def test_matyas(self):
        def matyas(x, y):
            z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
            return z

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        with train():
            z = matyas(x, y)
            z.backward()
        self.assertAlmostEqual(np.array(0.04), x.grad)
        self.assertAlmostEqual(np.array(0.04), y.grad)

    def test_goldstein_price(self):
        def goldstein(x, y):
            z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
                (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
            return z

        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        with train():
            z = goldstein(x, y)
            z.backward()
        self.assertAlmostEqual(np.array(-5376.0), x.grad)
        self.assertAlmostEqual(np.array(8064.0), y.grad)
