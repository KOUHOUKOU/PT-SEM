import argparse
import csv
import heapq
import math
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln,logsumexp


VARIABLES=["FOUL","FTA","FTM","MISS_FG","REB"]
EPS=1e-10
EXPECTED_DAGS_5=29281


def poisson_logpmf(k,lam):
    lam=np.maximum(np.asarray(lam,dtype=float),EPS)
    return -lam+k*np.log(lam)-gammaln(k+1.0)


def nb_logpmf_scalar(e,mu,r):
    if mu<=0.0 or r<=0.0:
        return -np.inf
    p=r/(r+mu)
    p=min(max(p,EPS),1.0-EPS)
    return gammaln(e+r)-gammaln(r)-gammaln(e+1.0)+r*math.log(p)+e*math.log1p(-p)


def validate_count_frame(frame,path):
    missing=[v for v in VARIABLES if v not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing variables: {missing}")

    values=frame[VARIABLES].apply(pd.to_numeric,errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"{path} has nonnumeric values")
    if (values<0).any().any():
        raise ValueError(f"{path} has negative counts")
    if not np.allclose(values.to_numpy(),np.round(values.to_numpy())):
        raise ValueError(f"{path} has noninteger counts")
    if (values["FTM"]>values["FTA"]).any():
        raise ValueError(f"{path} violates FTM <= FTA")

    if {"game_id","period"}.issubset(frame.columns):
        duplicated=frame.duplicated(["game_id","period"])
        if duplicated.any():
            raise ValueError(f"{path} has duplicate game-quarter rows: {int(duplicated.sum())}")

    return values.astype("int64")


def covariance_initial(data,child,parents):
    y=data[:,child].astype(float)

    if not parents:
        return np.zeros(0),max(float(y.mean()),1e-3),max(float(y.var(ddof=1)),1e-3)

    mu=data.mean(axis=0)
    cov=np.cov(data.astype(float),rowvar=False,bias=False)
    cpp=cov[np.ix_(parents,parents)]
    cpi=cov[np.ix_(parents,[child])].reshape(-1)

    try:
        alpha=np.linalg.solve(cpp+np.eye(len(parents))*1e-9,cpi)
    except np.linalg.LinAlgError:
        alpha=np.zeros(len(parents))

    alpha=np.clip(alpha,1e-4,10.0)
    parent_mean=float(alpha@mu[parents])
    mu_eps=max(float(mu[child]-parent_mean),1e-3)
    var_eps=max(float(cov[child,child]-alpha@cpp@alpha-parent_mean),1e-3)

    return alpha,mu_eps,var_eps


def theta_from_params(params,x_parent,n_parents):
    if n_parents==0:
        return np.zeros(x_parent.shape[0],dtype=float)
    return x_parent@np.asarray(params[:n_parents],dtype=float)


def nll_poisson(params,y,x_parent):
    n_parents=x_parent.shape[1]
    theta=theta_from_params(params,x_parent,n_parents)
    lam_eps=params[n_parents]
    return -float(np.sum(poisson_logpmf(y,theta+lam_eps)))


def nll_nb(params,y,x_parent):
    n_parents=x_parent.shape[1]
    theta=theta_from_params(params,x_parent,n_parents)
    mu_eps=params[n_parents]
    r=params[n_parents+1]

    if mu_eps<=0.0 or r<=0.0:
        return np.inf

    max_y=int(y.max())
    nb_cache=np.asarray([nb_logpmf_scalar(e,mu_eps,r) for e in range(max_y+1)])

    total=0.0
    for yi in range(max_y+1):
        idx=np.where(y==yi)[0]
        if idx.size==0:
            continue
        e_vals=np.arange(yi+1)
        pois_terms=poisson_logpmf(yi-e_vals[:,None],theta[idx][None,:])
        total+=float(np.sum(logsumexp(nb_cache[e_vals,None]+pois_terms,axis=0)))

    return -total


def fit_local(data,child,parent_mask,family,maxiter):
    parents=[j for j in range(data.shape[1]) if parent_mask&(1<<j)]
    y=data[:,child].astype(float)
    x_parent=data[:,parents].astype(float) if parents else np.zeros((data.shape[0],0))

    alpha0,mu0,var0=covariance_initial(data,child,parents)
    alpha_bounds=[(0.0,10.0)]*len(parents)

    if family=="poisson":
        starts=[
            np.r_[alpha0,max(mu0,1e-3)],
            np.r_[np.maximum(alpha0*0.5,1e-4),max(float(y.mean()),1e-3)]
        ]
        bounds=alpha_bounds+[(1e-8,max(float(y.max())*3.0+10.0,10.0))]
        objective=lambda z:nll_poisson(z,y,x_parent)
        q=len(parents)+1

    elif family=="nb":
        r0=max(mu0*mu0/max(var0-mu0,1e-3),1.0) if var0>mu0 else 1e4
        starts=[
            np.r_[alpha0,max(mu0,1e-3),min(r0,1e6)],
            np.r_[alpha0,max(mu0,1e-3),2.0],
            np.r_[np.maximum(alpha0*0.5,1e-4),max(float(y.mean()),1e-3),10.0]
        ]
        bounds=alpha_bounds+[
            (1e-8,max(float(y.max())*5.0+10.0,10.0)),
            (1e-6,1e8)
        ]
        objective=lambda z:nll_nb(z,y.astype(int),x_parent)
        q=len(parents)+2

    else:
        raise ValueError(f"unknown family: {family}")

    best=None
    for start in starts:
        result=minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter":maxiter,"ftol":1e-8}
        )
        value=float(result.fun)
        if np.isfinite(value) and (best is None or value<best.fun):
            best=result

    if best is None:
        return math.inf,None

    bic=2.0*float(best.fun)+q*math.log(data.shape[0])
    fit={
        "parents":parents,
        "params":best.x.tolist(),
        "nll":float(best.fun),
        "success":bool(best.success)
    }
    return bic,fit


def fit_local_task(args):
    data,child,parent_mask,family,maxiter=args
    bic,fit=fit_local(data,child,parent_mask,family,maxiter)
    return child,parent_mask,bic,fit


def build_local_scores(data,family,maxiter,workers):
    d=data.shape[1]
    tasks=[]
    for child in range(d):
        for parent_mask in range(1<<d):
            if not (parent_mask&(1<<child)):
                tasks.append((data,child,parent_mask,family,maxiter))

    local=[{} for _ in range(d)]
    fits=[{} for _ in range(d)]

    if workers<=1:
        for task in tasks:
            child,parent_mask,bic,fit=fit_local_task(task)
            local[child][parent_mask]=bic
            fits[child][parent_mask]=fit
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(fit_local_task,task) for task in tasks]
            for future in as_completed(futures):
                child,parent_mask,bic,fit=future.result()
                local[child][parent_mask]=bic
                fits[child][parent_mask]=fit

    return local,fits


def is_acyclic(edge_mask,directed_edges,d):
    indeg=[0]*d
    children=[[] for _ in range(d)]

    for bit,(src,dst) in enumerate(directed_edges):
        if edge_mask&(1<<bit):
            children[src].append(dst)
            indeg[dst]+=1

    stack=[i for i in range(d) if indeg[i]==0]
    seen=0

    while stack:
        node=stack.pop()
        seen+=1
        for child in children[node]:
            indeg[child]-=1
            if indeg[child]==0:
                stack.append(child)

    return seen==d


def enumerate_dags(local,top_k):
    d=len(local)
    directed_edges=[(src,dst) for src in range(d) for dst in range(d) if src!=dst]

    heap=[]
    n_acyclic=0
    n_finite=0

    for edge_mask in range(1<<len(directed_edges)):
        if not is_acyclic(edge_mask,directed_edges,d):
            continue

        n_acyclic+=1
        parent_masks=[0]*d

        for bit,(src,dst) in enumerate(directed_edges):
            if edge_mask&(1<<bit):
                parent_masks[dst]|=1<<src

        bic=sum(local[v][parent_masks[v]] for v in range(d))
        if not np.isfinite(bic):
            continue

        n_finite+=1

        if len(heap)<top_k:
            heapq.heappush(heap,(-float(bic),edge_mask,tuple(parent_masks)))
        elif bic<-heap[0][0]:
            heapq.heapreplace(heap,(-float(bic),edge_mask,tuple(parent_masks)))

    ranked=[]
    for neg_bic,edge_mask,parent_masks in sorted(heap,reverse=True):
        edges=[]
        for bit,(src,dst) in enumerate(directed_edges):
            if edge_mask&(1<<bit):
                edges.append((VARIABLES[src],VARIABLES[dst]))
        ranked.append((-neg_bic,sorted(edges),parent_masks))

    return n_acyclic,n_finite,ranked


def format_edges(edges):
    return ", ".join(f"{a}->{b}" for a,b in edges) or "(none)"


def mask_to_parent_names(mask):
    return ";".join(VARIABLES[j] for j in range(len(VARIABLES)) if mask&(1<<j))


def write_local_scores(path,family,local,fits):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["family","child","parent_set","bic","fit"])
        for child,table in enumerate(local):
            for parent_mask,bic in sorted(table.items()):
                writer.writerow([
                    family,
                    VARIABLES[child],
                    mask_to_parent_names(parent_mask),
                    f"{bic:.12g}",
                    fits[child][parent_mask]
                ])


def infer_season(frame,path):
    if "season" in frame.columns and frame["season"].nunique()==1:
        return str(frame["season"].iloc[0])
    stem=Path(path).stem
    for token in stem.split("_"):
        if "-" in token and len(token)>=7:
            return token
    return stem


def run_one_csv(path,args):
    frame=pd.read_csv(path)
    season=infer_season(frame,path)
    data_frame=validate_count_frame(frame,path)
    data=data_frame[VARIABLES].to_numpy(dtype="int64")

    print(f"[{season}] input={path}")
    print(f"[{season}] n={len(data)} means="+", ".join(
        f"{name}={data[:,i].mean():.3f}" for i,name in enumerate(VARIABLES)
    ),flush=True)

    row={"season":season,"n":len(data),"input":str(path)}

    for family in args.families:
        print(f"[{season}] fitting {family}",flush=True)
        local,fits=build_local_scores(data,family,args.maxiter,args.workers)

        if args.write_local:
            local_path=args.outputs_dir/"local_scores"/f"local_scores_{season}_{family}.csv"
            write_local_scores(local_path,family,local,fits)

        n_acyclic,n_finite,ranked=enumerate_dags(local,args.top_k)

        if n_acyclic!=EXPECTED_DAGS_5:
            raise RuntimeError(f"expected {EXPECTED_DAGS_5} acyclic DAGs over five nodes, got {n_acyclic}")
        if n_finite!=EXPECTED_DAGS_5:
            raise RuntimeError(f"{season} {family} has only {n_finite} finite DAG scores")

        best_bic,best_edges,best_parent_masks=ranked[0]
        row[f"{family}_bic"]=best_bic
        row[f"{family}_edges"]=format_edges(best_edges)
        row[f"{family}_parent_masks"]="|".join(map(str,best_parent_masks))
        row[f"{family}_n_acyclic"]=n_acyclic
        row[f"{family}_n_finite"]=n_finite

        print(f"  {family}: BIC={best_bic:,.3f} {format_edges(best_edges)}",flush=True)

    return row


def collect_inputs(args):
    if args.input:
        return [args.input]

    paths=sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {args.input_dir / args.pattern}")

    return paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/rebound_control_study/game-quarter"))
    parser.add_argument("--pattern",default="game-quarter_counts_*_drawn_missfg_reb.csv")
    parser.add_argument("--outputs-dir",type=Path,default=Path("outputs/five_var_recovery_clean"))
    parser.add_argument("--families",nargs="+",default=["poisson","nb"],choices=["poisson","nb"])
    parser.add_argument("--maxiter",type=int,default=300)
    parser.add_argument("--workers",type=int,default=max(1,min(6,(os.cpu_count() or 2)-1)))
    parser.add_argument("--top-k",type=int,default=10)
    parser.add_argument("--write-local",action="store_true")
    args=parser.parse_args()

    args.outputs_dir.mkdir(parents=True,exist_ok=True)

    rows=[]
    for path in collect_inputs(args):
        rows.append(run_one_csv(path,args))

    out_path=args.outputs_dir/"five_var_recovery_results.csv"
    pd.DataFrame(rows).to_csv(out_path,index=False)
    print(f"results written to {out_path}")


if __name__=="__main__":
    main()