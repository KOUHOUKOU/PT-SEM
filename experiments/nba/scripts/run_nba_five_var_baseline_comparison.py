import argparse
import ast
import importlib
import inspect
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


VARIABLES=["FOUL","FTA","FTM","MISS_FG","REB"]

RULE_EDGES={
    ("FOUL","FTA"),
    ("FTA","FTM"),
    ("FTA","REB"),
    ("MISS_FG","REB"),
}

EPS=1e-10


def format_edges(edges):
    edges=sorted(edges)
    return ", ".join(f"{a}->{b}" for a,b in edges) if edges else "(none)"


def parse_edge_string(text):
    if text is None or pd.isna(text):
        return set()
    text=str(text).strip()
    if not text or text=="(none)":
        return set()
    edges=set()
    for part in text.split(","):
        part=part.strip()
        if "->" not in part:
            continue
        a,b=part.split("->",1)
        a=a.strip()
        b=b.strip()
        if a in VARIABLES and b in VARIABLES and a!=b:
            edges.add((a,b))
    return edges


def skeleton(edges):
    return {frozenset((a,b)) for a,b in edges if a!=b}


def score_sets(pred,true):
    pred=set(pred)
    true=set(true)
    tp=len(pred&true)
    fp=len(pred-true)
    fn=len(true-pred)

    precision=tp/(tp+fp) if tp+fp>0 else 0.0
    recall=tp/(tp+fn) if tp+fn>0 else 0.0
    f1=2.0*precision*recall/(precision+recall) if precision+recall>0 else 0.0

    return precision,recall,f1,tp,fp,fn


def evaluate_graph(pred_edges):
    skel_precision,skel_recall,skel_f1,skel_tp,skel_fp,skel_fn=score_sets(
        skeleton(pred_edges),
        skeleton(RULE_EDGES),
    )
    dir_precision,dir_recall,dir_f1,dir_tp,dir_fp,dir_fn=score_sets(
        pred_edges,
        RULE_EDGES,
    )

    return {
        "pred_edges":format_edges(pred_edges),
        "n_pred_edges":len(pred_edges),
        "skeleton_precision":skel_precision,
        "skeleton_recall":skel_recall,
        "skeleton_f1":skel_f1,
        "skeleton_tp":skel_tp,
        "skeleton_fp":skel_fp,
        "skeleton_fn":skel_fn,
        "directed_precision":dir_precision,
        "directed_recall":dir_recall,
        "directed_f1":dir_f1,
        "directed_tp":dir_tp,
        "directed_fp":dir_fp,
        "directed_fn":dir_fn,
        "exact_recovery":set(pred_edges)==set(RULE_EDGES),
    }


def season_from_frame_or_path(frame,path):
    if "season" in frame.columns and frame["season"].nunique()==1:
        return str(frame["season"].iloc[0])
    stem=Path(path).stem
    for token in stem.split("_"):
        if "-" in token:
            return token
    return stem


def validate_data_frame(frame,path):
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
        if frame.duplicated(["game_id","period"]).any():
            raise ValueError(f"{path} has duplicate game-quarter rows")

    return values.astype("int64")


def load_count_files(input_dir,pattern):
    paths=sorted(Path(input_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {Path(input_dir)/pattern}")

    data_by_season={}
    info_by_season={}

    for path in paths:
        frame=pd.read_csv(path)
        season=season_from_frame_or_path(frame,path)
        values=validate_data_frame(frame,path)
        data=values[VARIABLES].to_numpy(dtype="int64")

        data_by_season[season]=data
        info_by_season[season]={
            "season":season,
            "n":len(data),
            "input":str(path),
            **{f"mean_{v}":float(values[v].mean()) for v in VARIABLES},
        }

    return data_by_season,info_by_season


def poisson_nll_glm(params,y,x):
    eta=params[0]+x@params[1:] if x.shape[1]>0 else np.full(len(y),params[0])
    eta=np.clip(eta,-30.0,30.0)
    lam=np.exp(eta)
    return float(np.sum(lam-y*eta+gammaln(y+1.0)))


def fit_poisson_glm_bic(y,x,maxiter):
    y=np.asarray(y,dtype=float)
    x=np.asarray(x,dtype=float)

    n=len(y)
    p=x.shape[1]
    y_mean=max(float(y.mean()),1e-6)

    if p>0:
        x_mean=x.mean(axis=0)
        x_std=x.std(axis=0)
        x_std=np.where(x_std<EPS,1.0,x_std)
        x_scaled=(x-x_mean)/x_std
    else:
        x_scaled=np.zeros((n,0))

    start=np.zeros(p+1)
    start[0]=math.log(y_mean)

    result=minimize(
        lambda z:poisson_nll_glm(z,y,x_scaled),
        start,
        method="L-BFGS-B",
        options={"maxiter":maxiter,"ftol":1e-9},
    )

    nll=float(result.fun)
    q=p+1
    bic=2.0*nll+q*math.log(n)

    return bic,result


def all_parent_subsets(predecessors,max_parents):
    predecessors=list(predecessors)
    upper=len(predecessors) if max_parents is None else min(len(predecessors),max_parents)
    subsets=[]
    for r in range(upper+1):
        for sub in combinations(predecessors,r):
            subsets.append(list(sub))
    return subsets


def ods_scores(data):
    scores={}
    for j,name in enumerate(VARIABLES):
        x=data[:,j].astype(float)
        mu=float(x.mean())
        var=float(x.var(ddof=1))
        ratio=var/max(mu,EPS)
        overdisp=(var-mu)/max(mu,EPS)
        scores[name]={
            "mean":mu,
            "var":var,
            "var_mean_ratio":ratio,
            "overdispersion":overdisp,
        }
    return scores


def run_poisson_ods_baseline(data,maxiter,max_parents):
    scores=ods_scores(data)

    order=sorted(
        VARIABLES,
        key=lambda name:(scores[name]["var_mean_ratio"],name),
    )

    edges=set()
    local_bics={}

    for child_name in order:
        child=VARIABLES.index(child_name)
        predecessors=[VARIABLES.index(v) for v in order[:order.index(child_name)]]

        best_bic=None
        best_parents=None

        for parents in all_parent_subsets(predecessors,max_parents):
            y=data[:,child]
            x=data[:,parents] if parents else np.zeros((data.shape[0],0))
            bic,_=fit_poisson_glm_bic(y,x,maxiter)

            if best_bic is None or bic<best_bic:
                best_bic=bic
                best_parents=parents

        local_bics[child_name]=best_bic
        for p in best_parents:
            edges.add((VARIABLES[p],child_name))

    return edges,order,scores,local_bics


def adjacency_to_edges(adj,transpose=False):
    arr=np.asarray(adj)
    if arr.ndim!=2 or arr.shape[0]!=arr.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape {arr.shape}")
    if arr.shape[0]!=len(VARIABLES):
        raise ValueError(f"adjacency matrix size {arr.shape[0]} does not match {len(VARIABLES)} variables")

    if transpose:
        arr=arr.T

    edges=set()
    for i,a in enumerate(VARIABLES):
        for j,b in enumerate(VARIABLES):
            if i!=j and float(arr[i,j])!=0.0:
                edges.add((a,b))
    return edges


def read_pbscm_adj_from_csv(path,transpose=False):
    frame=pd.read_csv(path,index_col=0)
    try:
        arr=frame.loc[VARIABLES,VARIABLES].to_numpy()
    except Exception:
        frame=pd.read_csv(path,header=None)
        arr=frame.to_numpy()
    return adjacency_to_edges(arr,transpose=transpose)


def try_call_pbscm_function(func,*args):
    try:
        return func(*args)
    except TypeError:
        return None
    except Exception:
        return None


def normalize_pbscm_result_to_adj(result):
    if result is None:
        return None

    if isinstance(result,tuple) or isinstance(result,list):
        for item in result:
            adj=normalize_pbscm_result_to_adj(item)
            if adj is not None:
                return adj
        return None

    if isinstance(result,pd.DataFrame):
        return result.to_numpy()

    if isinstance(result,np.ndarray):
        return result

    if hasattr(result,"dag"):
        return normalize_pbscm_result_to_adj(result.dag)

    if hasattr(result,"causal_graph"):
        return normalize_pbscm_result_to_adj(result.causal_graph)

    if hasattr(result,"adjacency_matrix"):
        return normalize_pbscm_result_to_adj(result.adjacency_matrix)

    return None


def run_pbscm_auto(data,pbscm_root,transpose=False):
    if pbscm_root:
        root=str(Path(pbscm_root).resolve())
        if root not in sys.path:
            sys.path.insert(0,root)

    module_names=[
        "PB_SCM",
        "PBSCM",
        "pbscm",
        "src.PB_SCM",
        "src.PBSCM",
        "model.PB_SCM",
        "models.PB_SCM",
    ]

    errors=[]

    for module_name in module_names:
        try:
            module=importlib.import_module(module_name)
        except Exception as e:
            errors.append(f"{module_name}: import failed: {e}")
            continue

        if hasattr(module,"Hill_Climb_search") and hasattr(module,"learning_causal_direction"):
            hill=getattr(module,"Hill_Climb_search")
            orient=getattr(module,"learning_causal_direction")

            for arr in [data,data.T]:
                skeleton_result=try_call_pbscm_function(hill,arr)
                if skeleton_result is None:
                    continue
                dag_result=try_call_pbscm_function(orient,arr,skeleton_result)
                adj=normalize_pbscm_result_to_adj(dag_result)
                if adj is not None:
                    return adjacency_to_edges(adj,transpose=transpose),f"{module_name}: Hill_Climb_search + learning_causal_direction"

        for class_name in ["PBSCM","PB_SCM","PBSCM_PGF"]:
            if hasattr(module,class_name):
                cls=getattr(module,class_name)
                for arr in [data,data.T]:
                    try:
                        model=cls(arr)
                        if hasattr(model,"learn"):
                            model.learn()
                        elif hasattr(model,"fit"):
                            model.fit()
                        adj=normalize_pbscm_result_to_adj(model)
                        if adj is not None:
                            return adjacency_to_edges(adj,transpose=transpose),f"{module_name}.{class_name}"
                    except Exception as e:
                        errors.append(f"{module_name}.{class_name}: {e}")

    raise RuntimeError("PB-SCM auto call failed. Tried modules: "+" | ".join(errors[:8]))


def load_ptsem_results(path):
    path=Path(path)
    if not path.exists():
        return {}

    frame=pd.read_csv(path)
    result={}

    for _,row in frame.iterrows():
        season=str(row["season"])
        result.setdefault(season,{})

        if "poisson_edges" in frame.columns:
            result[season]["PT-SEM (Poisson)"]=parse_edge_string(row["poisson_edges"])
        if "nb_edges" in frame.columns:
            result[season]["PT-SEM (NB)"]=parse_edge_string(row["nb_edges"])

    return result


def append_eval(rows,season,method,edges,n,extra=None):
    metrics=evaluate_graph(edges)
    row={
        "season":season,
        "method":method,
        "n":n,
        **metrics,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def summarize_results(detail_frame):
    rows=[]
    for method,sub in detail_frame.groupby("method",sort=False):
        rows.append({
            "method":method,
            "n_seasons":int(sub["season"].nunique()),
            "mean_skeleton_precision":float(sub["skeleton_precision"].mean()),
            "mean_skeleton_recall":float(sub["skeleton_recall"].mean()),
            "mean_skeleton_f1":float(sub["skeleton_f1"].mean()),
            "mean_directed_precision":float(sub["directed_precision"].mean()),
            "mean_directed_recall":float(sub["directed_recall"].mean()),
            "mean_directed_f1":float(sub["directed_f1"].mean()),
            "exact_recovery_count":int(sub["exact_recovery"].sum()),
            "exact_recovery_rate":float(sub["exact_recovery"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/rebound_control_study/game-quarter"))
    parser.add_argument("--pattern",default="game-quarter_counts_*_drawn_missfg_reb.csv")
    parser.add_argument("--ptsem-results",type=Path,default=Path("outputs/five_var_recovery_clean/five_var_recovery_results.csv"))
    parser.add_argument("--outputs-dir",type=Path,default=Path("outputs/nba_five_var_baseline_comparison"))
    parser.add_argument("--maxiter",type=int,default=300)
    parser.add_argument("--max-parents",type=int,default=None)
    parser.add_argument("--run-pbscm",action="store_true")
    parser.add_argument("--pbscm-root",type=Path)
    parser.add_argument("--pbscm-adj-dir",type=Path)
    parser.add_argument("--pbscm-adj-pattern",default="pbscm_adj_{season}.csv")
    parser.add_argument("--transpose-pbscm-adj",action="store_true")
    args=parser.parse_args()

    args.outputs_dir.mkdir(parents=True,exist_ok=True)

    data_by_season,info_by_season=load_count_files(args.input_dir,args.pattern)
    ptsem_results=load_ptsem_results(args.ptsem_results)

    detail_rows=[]
    ods_diag_rows=[]

    for season,data in data_by_season.items():
        n=data.shape[0]
        print(f"[{season}] n={n}",flush=True)

        if season in ptsem_results:
            for method,edges in ptsem_results[season].items():
                append_eval(
                    detail_rows,
                    season,
                    method,
                    edges,
                    n,
                    extra={"source":"ptsem_results_csv"},
                )
                print(f"  {method}: {format_edges(edges)}",flush=True)

        ods_edges,ods_order,ods_score_table,ods_local_bics=run_poisson_ods_baseline(
            data,
            maxiter=args.maxiter,
            max_parents=args.max_parents,
        )
        append_eval(
            detail_rows,
            season,
            "Poisson DAG (ODS-style)",
            ods_edges,
            n,
            extra={
                "source":"implemented_ods_style_baseline",
                "order":" -> ".join(ods_order),
            },
        )
        print(f"  Poisson DAG (ODS-style): order={' -> '.join(ods_order)} edges={format_edges(ods_edges)}",flush=True)

        for name in VARIABLES:
            row={
                "season":season,
                "variable":name,
                "order":" -> ".join(ods_order),
                "local_bic":ods_local_bics.get(name,np.nan),
                **ods_score_table[name],
            }
            ods_diag_rows.append(row)

        if args.run_pbscm:
            pbscm_edges=None
            pbscm_source=None

            if args.pbscm_adj_dir is not None:
                adj_path=args.pbscm_adj_dir/args.pbscm_adj_pattern.format(season=season)
                if adj_path.exists():
                    pbscm_edges=read_pbscm_adj_from_csv(adj_path,transpose=args.transpose_pbscm_adj)
                    pbscm_source=str(adj_path)

            if pbscm_edges is None:
                try:
                    pbscm_edges,pbscm_source=run_pbscm_auto(
                        data,
                        args.pbscm_root,
                        transpose=args.transpose_pbscm_adj,
                    )
                except Exception as e:
                    print(f"  PB-SCM skipped: {e}",flush=True)

            if pbscm_edges is not None:
                append_eval(
                    detail_rows,
                    season,
                    "PB-SCM",
                    pbscm_edges,
                    n,
                    extra={"source":pbscm_source},
                )
                print(f"  PB-SCM: {format_edges(pbscm_edges)}",flush=True)

    detail=pd.DataFrame(detail_rows)
    summary=summarize_results(detail)
    info=pd.DataFrame(list(info_by_season.values()))
    ods_diag=pd.DataFrame(ods_diag_rows)

    detail_path=args.outputs_dir/"per_season_graph_recovery.csv"
    summary_path=args.outputs_dir/"method_summary.csv"
    info_path=args.outputs_dir/"season_input_summary.csv"
    ods_diag_path=args.outputs_dir/"ods_diagnostics.csv"

    detail.to_csv(detail_path,index=False)
    summary.to_csv(summary_path,index=False)
    info.to_csv(info_path,index=False)
    ods_diag.to_csv(ods_diag_path,index=False)

    print(f"written to {detail_path}")
    print(f"written to {summary_path}")
    print(f"written to {info_path}")
    print(f"written to {ods_diag_path}")
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()