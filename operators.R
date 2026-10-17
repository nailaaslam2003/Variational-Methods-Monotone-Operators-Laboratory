# Proximal operators and projections.
soft_threshold <- function(x, threshold) {
  if (threshold < 0) stop("threshold must be nonnegative")
  sign(x) * pmax(abs(x)-threshold, 0)
}
prox_l1 <- function(x, tau=1, weight=1) {
  if (tau < 0 || weight < 0) stop("tau and weight must be nonnegative")
  soft_threshold(x, tau*weight)
}
project_box <- function(x, lower=0, upper=1) {
  if (any(lower > upper)) stop("lower must not exceed upper")
  pmin(pmax(x,lower),upper)
}
