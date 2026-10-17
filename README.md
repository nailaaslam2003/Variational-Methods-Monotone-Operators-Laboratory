# VMOM-Lab: Variational Methods & Monotone Operators Laboratory

A reproducible Python + R project exploring proximal algorithms, nonsmooth convex optimization, and monotone operators. It includes source code, synthetic-data generation, tests, mathematical notes, and reproducible experiments.

## Research questions
1. How do ISTA and FISTA compare on sparse linear inverse problems?
2. How does the regularization parameter affect sparsity and reconstruction error?
3. How does soft-thresholding connect to resolvents of maximal monotone operators?
4. How do stopping tolerances and problem conditioning affect observed convergence?

## Model problem: LASSO
\[
\min_x F(x)=\frac{1}{2n}\|Ax-y\|_2^2+\lambda\|x\|_1.
\]
Here \(\nabla f(x)=A^\top(Ax-y)/n\), with \(L=\|A\|_2^2/n\). The proximal map of the \(\ell_1\) penalty is soft-thresholding:
\[
\operatorname{prox}_{\tau\lambda\|\cdot\|_1}(z)=\operatorname{sign}(z)\max(|z|-\tau\lambda,0).
\]

## Structure
- `src/vmomlab/`: Python proximal operators and first-order solvers
- `R/`: independent base-R implementations and tests
- `examples/`: deterministic end-to-end experiment
- `tests/`: Python unit tests
- `docs/`: theory, experiment protocol, roadmap
- `data/`: synthetic data policy

## Python setup
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python examples/run_lasso_experiment.py
python -m unittest discover -s tests -v
```
Outputs are written to `outputs/` (CSV trajectory, PNG convergence plot, text summary).

## R setup
Requires R 4.1+ and base packages only.
```bash
Rscript R/run_lasso_experiment.R
Rscript R/test_algorithms.R
```
Outputs are written to `outputs-r/`.

## Reproducibility and limitations
All data are generated locally from documented distributions and explicit seeds. R and NumPy use different random-number generators, so their data will not be identical. The implementations are reference/learning code, not a replacement for specialist solvers. Fixed-step algorithms rely on convexity and step-size assumptions. The plotted objective-gap proxy is not a certified optimality gap.



