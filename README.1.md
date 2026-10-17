# Data policy

No external dataset is required. Experiments generate synthetic Gaussian design matrices, sparse coefficients, and Gaussian noise from documented distributions and explicit seeds. Python uses NumPy seed 2026; R uses `set.seed(2026)`. Because their random-number generators differ, samples will not be byte-identical. Outputs are generated on demand rather than stored as opaque files.
