"""Compare ISTA and monotone FISTA on reproducible synthetic sparse data."""
from pathlib import Path
import csv, sys
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vmomlab.solvers import ista, fista

def generate_data(seed=2026,n=160,p=240,sparsity=12,noise_sd=.03):
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(n,p))
    truth=np.zeros(p); support=rng.choice(p,size=sparsity,replace=False)
    truth[support]=rng.normal(size=sparsity)
    return A,A@truth+rng.normal(scale=noise_sd,size=n),truth,support

def main():
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    A,y,truth,support=generate_data()
    lam=.025; opts=dict(lam=lam,max_iter=2500,tol=1e-9)
    a=ista(A,y,**opts); b=fista(A,y,**opts,monotone=True)
    with (out/"convergence.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["algorithm","iteration","objective","lambda","n","p","seed"])
        for name,r in [("ISTA",a),("FISTA-monotone",b)]:
            for i,v in enumerate(r["objective"],1):
                w.writerow([name,i,f"{v:.12g}",lam,A.shape[0],A.shape[1],2026])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    best=min(a["objective"][-1],b["objective"][-1])
    ax.semilogy(np.arange(1,len(a["objective"])+1),np.maximum(a["objective"]-best+1e-14,1e-14),label="ISTA")
    ax.semilogy(np.arange(1,len(b["objective"])+1),np.maximum(b["objective"]-best+1e-14,1e-14),label="FISTA (monotone)")
    ax.set(xlabel="Iteration",ylabel="Objective minus best final objective (proxy)",title="Sparse recovery: convergence comparison")
    ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/"convergence.png",dpi=160); plt.close(fig)
    summary={"ISTA_iterations":a["iterations"],"FISTA_iterations":b["iterations"],
      "ISTA_final_objective":float(a["objective"][-1]),"FISTA_final_objective":float(b["objective"][-1]),
      "truth_support_size":len(support),"estimated_support_size":int(np.sum(np.abs(b["x"])>1e-5)),
      "coefficient_error_l2":float(np.linalg.norm(b["x"]-truth))}
    (out/"summary.txt").write_text("\n".join(f"{k}: {v}" for k,v in summary.items())+"\\n")
    print("Experiment complete:",out); print((out/"summary.txt").read_text())

if __name__=="__main__": main()
