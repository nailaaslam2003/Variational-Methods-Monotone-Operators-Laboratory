import unittest
import numpy as np
from vmomlab.solvers import ista, fista, forward_backward, projected_gradient_vi

def sparse_problem(seed=7, n=100, p=60, sparsity=8):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n,p))
    truth = np.zeros(p); idx = rng.choice(p, size=sparsity, replace=False)
    truth[idx] = rng.normal(size=sparsity)
    return A, A@truth + .02*rng.normal(size=n), truth

class SolverTests(unittest.TestCase):
    def test_fista_competitive_objective(self):
        A,y,_ = sparse_problem()
        a = ista(A,y,lam=.03,max_iter=1500)
        b = fista(A,y,lam=.03,max_iter=1500)
        self.assertLessEqual(b["objective"][-1], a["objective"][-1]+2e-4)
        self.assertTrue(np.all(np.isfinite(b["x"])))
    def test_monotone_fista_history(self):
        A,y,_ = sparse_problem()
        b = fista(A,y,lam=.02,max_iter=300,monotone=True)
        self.assertTrue(np.all(np.isfinite(b["objective"])))
        self.assertLessEqual(b["objective"][-1], b["objective"][0])
        self.assertTrue(np.all(np.diff(b["objective"]) <= 1e-10))
    def test_forward_backward_quadratic(self):
        Q=np.diag([1.,2.,3.]); b=np.array([1.,-2.,3.])
        result=forward_backward(lambda x: Q@x-b, lambda v,s: v, np.zeros(3), .2)
        np.testing.assert_allclose(result["x"], np.linalg.solve(Q,b), atol=2e-5)
    def test_projected_vi(self):
        a=np.array([-.5,.3,1.7])
        result=projected_gradient_vi(lambda x:x-a, np.array([.5,.5,.5]), step=.5)
        np.testing.assert_allclose(result["x"], [0.,.3,1.], atol=2e-5)

if __name__ == "__main__":
    unittest.main()
