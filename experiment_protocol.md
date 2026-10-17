# Experiment protocol

1. Generate a Gaussian design matrix with seed 2026.
2. Set 12 randomly selected coefficients to nonzero values.
3. Add Gaussian observation noise with standard deviation 0.03.
4. Solve the same LASSO objective using ISTA and monotone FISTA.
5. Save the full objective trajectory, iteration counts, support size, and coefficient error.

## Suggested extensions
- Sweep \(\lambda\) and plot sparsity against prediction/reconstruction error.
- Vary noise level and compare support precision/recall.
- Compare correlated versus nearly orthogonal design matrices.
- Add backtracking line search or Douglas–Rachford splitting.
- Compare forward-backward and projected-gradient methods on strongly convex examples.
- Use identical exported data to validate Python and R implementations.

Record seed, dimensions, sparsity, noise, regularization, tolerance, iteration cap, and stopping rule. Do not infer a theoretical convergence rate from a single plot.
