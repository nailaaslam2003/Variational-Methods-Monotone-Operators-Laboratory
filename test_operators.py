import unittest
import numpy as np
from vmomlab.operators import soft_threshold, prox_l1, prox_group_lasso, project_box

class OperatorTests(unittest.TestCase):
    def test_soft_threshold(self):
        np.testing.assert_allclose(soft_threshold([-2., -.2, .5, 3.], 1.), [-1., 0., 0., 2.])
    def test_prox_l1(self):
        np.testing.assert_allclose(prox_l1([-2., 0., 3.], .5, 2), [-1., 0., 2.])
    def test_group_lasso(self):
        out = prox_group_lasso([3., 4., 1.], [[0, 1]], tau=1.)
        np.testing.assert_allclose(out, [2.4, 3.2, 1.])
    def test_project_box(self):
        np.testing.assert_allclose(project_box([-1., .5, 2.]), [0., .5, 1.])
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            prox_group_lasso(np.ones(3), [[0, 1], [1, 2]])

if __name__ == "__main__":
    unittest.main()
