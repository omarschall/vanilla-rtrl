import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
from utils import *

class Test_Utils(unittest.TestCase):
    """Tests methods from the utils.py module."""

    def test_norm(self):
        """Verifies that norm asserts that an array of 1s of shape (2, 2, 2)
        returns the square root of 8."""

        x = np.ones((2, 2, 2))
        self.assertTrue(norm(x), np.sqrt(8))

    def test_split_weight_matrix(self):
        """Verifies that split_weight_matrix splits an array as intended."""

        x = np.ones((2, 10))
        x_list = split_weight_matrix(x, [3, 3, 4], axis=1)
        self.assertTrue(x_list[0].shape == (2,3))
        self.assertTrue(x_list[1].shape == (2,3))
        self.assertTrue(x_list[2].shape == (2,4))

        x = np.ones((10, 2))
        x_list = split_weight_matrix(x, [3, 3, 4], axis=0)
        self.assertTrue(x_list[0].shape == (3,2))
        self.assertTrue(x_list[1].shape == (3,2))
        self.assertTrue(x_list[2].shape == (4,2))

    def test_rectangular_filter(self):
        """Verifies that the rectangular convolution returns 0s for a simple
        sequence of alternating 1s and -1s with filter_size 2."""

        x = np.array([1, -1]*10)
        self.assertTrue((rectangular_filter(x, filter_size=2) == 0).all())

    def test_classification_accuracy(self):
        """Verifies that """

    def test_regetattr(self):
        """Verifies regetattr in a simple nested object case."""

        class Foo:
            def __init__(self):
                self.y = 2
        class Bar:
            def __init__(self):
                self.x = Foo()

        bar = Bar()

        self.assertEqual(rgetattr(bar, 'x.y'), 2)

    def test_triangular_integer_decomposition(self):

        idx = 0
        n, r = triangular_integer_decomposition(idx)
        self.assertEqual(n, 0)
        self.assertEqual(r, 0)

        idx = 11522401
        n, r = triangular_integer_decomposition(idx)
        self.assertEqual(n, 4800)
        self.assertEqual(r, 1)
        self.assertEqual(n*(n+1)/2 + r, idx)

        idx = 11527199
        n, r = triangular_integer_decomposition(idx)
        self.assertEqual(n, 4800)
        self.assertEqual(r, 4799)
        self.assertEqual(n * (n + 1) / 2 + r, idx)

if __name__ == '__main__':
    unittest.main()