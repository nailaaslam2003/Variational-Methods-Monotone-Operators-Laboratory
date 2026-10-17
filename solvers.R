# ISTA/FISTA for min_x ||Ax-y||^2/(2n) + lambda*||x||_1.
lasso_objective <- function(A,y,x,lambda) {
  r <- as.vector(A %*% x-y); sum(r^2)/(2*nrow(A))+lambda*sum(abs(x))
}
spectral_lipschitz <- function(A) max(as.numeric(norm(A,type="2")^2/nrow(A)),1e-12)
ista <- function(A,y,lambda=.1,max_iter=2000,tol=1e-7,x0=NULL) {
  if (lambda<0 || max_iter<1 || tol<0) stop("invalid parameter")
  A<-as.matrix(A); y<-as.numeric(y)
  if(nrow(A)!=length(y)) stop("A row count must match y")
  p<-ncol(A); x<-if(is.null(x0)) rep(0,p) else as.numeric(x0)
  if(length(x)!=p) stop("invalid x0")
  L<-spectral_lipschitz(A); hist<-numeric(0)
  for(k in seq_len(max_iter)) {
    grad<-as.vector(crossprod(A,A%*%x-y)/nrow(A))
    xn<-prox_l1(x-grad/L,tau=lambda/L)
    hist<-c(hist,lasso_objective(A,y,xn,lambda))
    if(sqrt(sum((xn-x)^2))<=tol*max(1,sqrt(sum(x^2)))) {x<-xn;break}
    x<-xn
  }
  list(x=x,objective=hist,iterations=length(hist),L=L)
}
fista <- function(A,y,lambda=.1,max_iter=2000,tol=1e-7,x0=NULL,monotone=FALSE) {
  if(lambda<0 || max_iter<1 || tol<0) stop("invalid parameter")
  A<-as.matrix(A); y<-as.numeric(y)
  if(nrow(A)!=length(y)) stop("A row count must match y")
  p<-ncol(A); x<-if(is.null(x0)) rep(0,p) else as.numeric(x0)
  if(length(x)!=p) stop("invalid x0")
  z<-x; t<-1; L<-spectral_lipschitz(A); hist<-numeric(0)
  old<-lasso_objective(A,y,x,lambda)
  for(k in seq_len(max_iter)) {
    grad<-as.vector(crossprod(A,A%*%z-y)/nrow(A))
    cand<-prox_l1(z-grad/L,tau=lambda/L); val<-lasso_objective(A,y,cand,lambda)
    if(monotone && val>old+1e-12) {
      grad<-as.vector(crossprod(A,A%*%x-y)/nrow(A))
      cand<-prox_l1(x-grad/L,tau=lambda/L); val<-lasso_objective(A,y,cand,lambda)
      tn<-1; zn<-cand
    } else {
      tn<-(1+sqrt(1+4*t^2))/2; zn<-cand+((t-1)/tn)*(cand-x)
    }
    hist<-c(hist,val)
    if(sqrt(sum((cand-x)^2))<=tol*max(1,sqrt(sum(x^2)))) {x<-cand;break}
    x<-cand; z<-zn; t<-tn; old<-val
  }
  list(x=x,objective=hist,iterations=length(hist),L=L)
}
