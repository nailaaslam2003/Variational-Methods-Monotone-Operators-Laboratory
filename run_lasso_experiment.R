source("R/operators.R"); source("R/solvers.R")
set.seed(2026)
n<-160; p<-240; sparsity<-12; noise_sd<-.03
A<-matrix(rnorm(n*p),n,p); truth<-rep(0,p)
support<-sample(seq_len(p),sparsity); truth[support]<-rnorm(sparsity)
y<-as.vector(A%*%truth+rnorm(n,sd=noise_sd)); lambda<-.025
a<-ista(A,y,lambda,max_iter=2500,tol=1e-9)
b<-fista(A,y,lambda,max_iter=2500,tol=1e-9,monotone=TRUE)
dir.create("outputs-r",showWarnings=FALSE)
df<-rbind(data.frame(algorithm="ISTA",iteration=seq_along(a$objective),objective=a$objective),
          data.frame(algorithm="FISTA-monotone",iteration=seq_along(b$objective),objective=b$objective))
df$lambda<-lambda; df$n<-n; df$p<-p; df$seed<-2026
write.csv(df,"outputs-r/convergence.csv",row.names=FALSE)
png("outputs-r/convergence.png",width=1000,height=650)
best<-min(tail(a$objective,1),tail(b$objective,1))
plot(seq_along(a$objective),pmax(a$objective-best+1e-14,1e-14),type="l",log="y",
 xlab="Iteration",ylab="Objective minus best final objective (proxy)",main="Sparse recovery convergence")
lines(seq_along(b$objective),pmax(b$objective-best+1e-14,1e-14),lty=2)
legend("topright",c("ISTA","FISTA (monotone)"),lty=c(1,2),bty="n"); grid(); dev.off()
s<-c(paste("ISTA iterations:",a$iterations),paste("FISTA iterations:",b$iterations),
 paste("ISTA final objective:",tail(a$objective,1)),paste("FISTA final objective:",tail(b$objective,1)),
 paste("True support size:",length(support)),paste("Estimated support size:",sum(abs(b$x)>1e-5)),
 paste("Coefficient L2 error:",sqrt(sum((b$x-truth)^2))))
writeLines(s,"outputs-r/summary.txt"); cat(paste(s,collapse="\n"),"\n")
