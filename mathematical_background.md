# Mathematical background

## Composite convex minimization
For \(\min_x f(x)+g(x)\), with \(f\) convex differentiable and \(\nabla f\) L-Lipschitz, and \(g\) proper lower-semicontinuous convex, forward-backward splitting is
\[
x_{k+1}=\operatorname{prox}_{\gamma g}(x_k-\gamma\nabla f(x_k)).
\]
The proximal map is
\[
\operatorname{prox}_{\gamma g}(v)=\arg\min_x\{g(x)+(2\gamma)^{-1}\|x-v\|^2\}.
\]

## LASSO and proximal gradient
For \(f(x)=\|Ax-y\|^2/(2n)\), \(g(x)=\lambda\|x\|_1\),
\[
\nabla f(x)=A^\top(Ax-y)/n,\quad L=\|A\|_2^2/n.
\]
ISTA uses
\[
x_{k+1}=\mathcal S_{\gamma\lambda}(x_k-\gamma A^\top(Ax_k-y)/n).
\]
Under standard assumptions, ISTA has an \(O(1/k)\) objective convergence rate; FISTA achieves an \(O(1/k^2)\) worst-case rate for the standard convex composite setting. Acceleration does not ensure every objective value decreases, which motivates the optional monotone safeguard.

## Monotone operators and resolvents
An operator \(T\) is monotone if \(\langle u-v,x-y\rangle\ge0\) whenever \(u\in T(x)\) and \(v\in T(y)\). For proper l.s.c. convex \(g\), its subdifferential \(\partial g\) is maximal monotone. The resolvent is \(J_{\gamma T}=(I+\gamma T)^{-1}\), and
\[
\operatorname{prox}_{\gamma g}=J_{\gamma\partial g}.
\]
Equivalently, \(x=\operatorname{prox}_{\gamma g}(z)\) iff \(0\in\partial g(x)+\gamma^{-1}(x-z)\).

## Variational inequalities
For closed convex \(C\), find \(x^*\in C\) such that \(\langle F(x^*),x-x^*\rangle\ge0\) for all \(x\in C\). A projected iteration is \(x_{k+1}=P_C(x_k-\gamma F(x_k))\). Convergence requires suitable assumptions on monotonicity, Lipschitz behavior, and step size; the implementation is a minimal demonstration rather than a universal solver.

## Interpretation
The plotted objective-gap proxy subtracts the best final objective among methods; it is not a certified primal gap unless the optimum is known. Low objective does not guarantee exact support recovery.
