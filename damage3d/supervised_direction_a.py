"""Standalone supervised Direction A; existing H3 modules are not modified."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .classes import DAMAGE_CLASSES
from .detectability import FusionModel
from .experiment_data import regions, provenance, validate_detector, match_rows, read_json, write_json
from .observations import iter_observations, probs_of
from .run_state import sha256_file

METHODS = ('mean','max','llr','pi_mean','pi_llr','censored')

def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=float)))

def logit(x):
    return np.log(x)-np.log1p(-x)

def aggregate(point,p,g,n,rho,g50,beta,clip):
    p=np.asarray(p,float)
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid probability')
    if np.any(~np.isfinite(g)) or np.any(g<=0):
        raise ValueError('Invalid GSD')
    count=np.bincount(point,minlength=n)
    lam=logit(np.clip(p,clip,1-clip))-logit(rho)
    z=(np.log(g50)-np.log(g))/beta
    logpi=-np.logaddexp(0.,-z)
    logone=-np.logaddexp(0.,z)
    pi=np.exp(logpi)
    b=lambda x: np.bincount(point,weights=x,minlength=n)
    mx=np.full(n,-np.inf); np.maximum.at(mx,point,p)
    den=b(pi)
    return dict(count=count,mean=b(p)/np.maximum(count,1),max=mx,
                llr=b(lam),pi_mean=b(pi*p)/np.maximum(den,np.finfo(float).tiny),
                pi_llr=b(pi*lam),censored=b(np.logaddexp(logpi+lam,logone)))

def probability(a,method,alpha,base):
    return a[method] if method in ('mean','max','pi_mean') else sigmoid(logit(base)+alpha*a[method])

def nll(y,s,w):
    s=np.clip(s,1e-7,1-1e-7)
    return float(np.average(-y*np.log(s)-(1-y)*np.log1p(-s),weights=w))

def metrics(y,s,w,t):
    y=np.asarray(y); w=np.asarray(w,float); pred=s>=t
    tp=float(w[(y==1)&pred].sum()); fp=float(w[(y==0)&pred].sum())
    fn=float(w[(y==1)&~pred].sum())
    order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]; ww=w[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    pos=float(w[y==1].sum()); neg=float(w[y==0].sum())
    ct=np.cumsum(ww*yy)[ends]; cw=np.cumsum(ww)[ends]
    ap=float(np.sum(np.diff(np.r_[0.,ct/pos])*(ct/cw))) if pos>0 else None
    return dict(n=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),
        positive_weight=pos,negative_weight=neg,nll=nll(y,s,w),
        brier=float(np.average((s-y)**2,weights=w)),ap=ap,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        iou=tp/(tp+fp+fn) if tp+fp+fn else None,threshold=float(t))

def cfgload(path):
    q=read_json(path)
    active=q['classes']
    if not active or len(set(active))!=len(active) or set(active)-set(DAMAGE_CLASSES):
        raise ValueError('Declare unique known active classes')
    for key in ('min_views','min_regions','sample_per_region','max_cached_observations'):
        if not isinstance(q[key],int) or q[key]<1: raise ValueError(key)
    if not q['g50_grid'] or not q['beta_grid'] or not q['alpha_grid'] or not q['threshold_grid']:
        raise ValueError('Empty grid')
    for key in ('g50_grid','beta_grid'):
        if any(not np.isfinite(x) or x<=0 for x in q[key]): raise ValueError(key)
    if any(not 0<x<=1 for x in q['alpha_grid']): raise ValueError('alpha_grid')
    if any(not 0<x<1 for x in q['threshold_grid']): raise ValueError('threshold_grid')
    return q

def context(a):
    q=cfgload(a.config); d=regions(a.regions,a.buffer); p=read_json(a.params)
    validate_detector(a.run,p)
    model=FusionModel.from_dict('llr',p)
    if any(abs(p['classes'][c].get('base_rate',p['classes'][c]['rho'])-p['classes'][c]['rho'])>1e-12 for c in q['classes']):
        raise ValueError('This protocol fixes posterior offset to rho too')
    return q,d,p,model,provenance(a.run,a.regions,a.config)

def report_support(d,rows,counts,q):
    report={}
    for split in ('C_fit','C_cal'):
        report[split]={}
        for c in q['classes']:
            j=list(DAMAGE_CLASSES).index(c); use=d['split'][rows]==split
            per={}
            for r in np.unique(d['region'][rows][use]):
                mask=use&(d['region'][rows]==r); y=d['labels'][rows,j]; ok=mask&(counts[:,j]>=q['min_views'])
                per[str(r)]={
                    'known_positive':int((mask&(y==1)).sum()),'known_negative':int((mask&(y==0)).sum()),
                    'unknown':int((mask&(y<0)).sum()),'usable_positive':int((ok&(y==1)).sum()),
                    'usable_negative':int((ok&(y==0)).sum()),
                    'positive_area':float(d['area'][rows][ok&(y==1)].sum()),
                    'negative_area':float(d['area'][rows][ok&(y==0)].sum())}
            report[split][c]=per
    return report

def preflight(a):
    q,d,p,model,prov=context(a)
    rows=np.flatnonzero(np.isin(d['split'],['C_fit','C_cal']))
    if not len(rows): raise ValueError('No development points')
    counts=np.zeros((len(d['source_index']),len(DAMAGE_CLASSES)),np.uint32)
    gmin=float('inf'); gmax=0.
    for obs in iter_observations(a.run,d['source_index'][rows]):
        g=model.view_gsd(obs['gsd_frontal'],obs['cos_theta'],obs['map_scale'])
        gmin=min(gmin,float(g.min())); gmax=max(gmax,float(g.max()))
        ix=match_rows(obs['source_index'],d); pp=probs_of(obs)
        if np.any(np.isinf(pp)) or np.any(np.isfinite(pp)&((pp<0)|(pp>1))): raise ValueError('Invalid p')
        for j in range(len(DAMAGE_CLASSES)): np.add.at(counts[:,j],ix,np.isfinite(pp[:,j]))
    report=report_support(d,rows,counts[rows],q); failures=[]
    for split in report:
        for c,rr in report[split].items():
            if sum(x['usable_positive'] for x in rr.values())==0: failures.append(f'{split}/{c}: no usable positives')
            if sum(x['usable_negative'] for x in rr.values())==0: failures.append(f'{split}/{c}: no usable negatives')
            nr=sum(x['usable_positive']+x['usable_negative']>0 for x in rr.values())
            if nr<q['min_regions']: failures.append(f'{split}/{c}: too few supported regions')
    write_json(a.out,dict(protocol='supervised_a_v1',provenance=prov,config=q,
        support=report,failures=failures,passed=not failures,gsd_range=[gmin,gmax],
        limitations=['No automatic visual GT audit','No local area-weight validation','T labels not summarized']))
    if failures: raise SystemExit('Preflight failed; inspect the saved JSON')

def select_fit(d, q):
    """Campionamento per regione e GT crack, con pesi di inclusione."""
    rng = np.random.default_rng(q["seed"])

    sampling = q.get("sampling", {})
    negative_limit = sampling.get("negative_per_region", 2000)
    unknown_limit = sampling.get("unknown_per_region", 1000)

    for value in (negative_limit, unknown_limit):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Le quote devono essere interi positivi.")

    crack_index = list(DAMAGE_CLASSES).index("crack")
    train = d["split"] == "C_fit"

    selected = []
    weights = []

    for region in np.unique(d["region"][train]):
        region_mask = train & (d["region"] == region)

        for label in (-1, 0, 1):
            candidates = np.flatnonzero(
                region_mask
                & (d["labels"][:, crack_index] == label)
            )

            population_size = len(candidates)
            if population_size == 0:
                continue

            if label == 1:
                # Tutti i positivi crack, senza duplicarli.
                picked = candidates
            else:
                limit = (
                    negative_limit if label == 0 else unknown_limit
                )
                sample_size = min(population_size, limit)

                picked = np.sort(
                    rng.choice(
                        candidates,
                        size=sample_size,
                        replace=False,
                    )
                )

            inclusion_probability = len(picked) / population_size

            selected.extend(picked.tolist())
            weights.extend(
                (
                    d["area"][picked] / inclusion_probability
                ).tolist()
            )

    if not selected:
        raise ValueError("Nessun punto selezionato in C_fit.")

    selected = np.asarray(selected, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    order = np.argsort(selected)
    return selected[order], weights[order]

def cache_fit(a,d,rows,model,q):
    chunks=[]; total=0
    for obs in iter_observations(a.run,d['source_index'][rows]):
        total+=len(obs)
        if total>q['max_cached_observations']:
            raise ValueError('Observation cache cap exceeded: reduce sample_per_region in a new config')
        chunks.append(obs)
    if not chunks: raise ValueError('No fit observations')
    obs=np.concatenate(chunks); order=np.lexsort((obs['camera'],obs['source_index'])); obs=obs[order]
    if np.any((obs['source_index'][1:]==obs['source_index'][:-1])&(obs['camera'][1:]==obs['camera'][:-1])):
        raise ValueError('Duplicate point-camera across chunks')
    ix=np.searchsorted(d['source_index'][rows],obs['source_index'])
    return ix,probs_of(obs),model.view_gsd(obs['gsd_frontal'],obs['cos_theta'],obs['map_scale'])

def fit(a):
    q,d,p,model,prov=context(a); pre=read_json(a.preflight)
    if not pre['passed'] or pre['provenance']!=prov or pre['config']!=q: raise ValueError('Preflight mismatch')
    rows,w=select_fit(d,q); ix,pp,g=cache_fit(a,d,rows,model,q); result={}
    for c in q['classes']:
        j=list(DAMAGE_CLASSES).index(c); valid=np.isfinite(pp[:,j]); ids=ix[valid]; pc=pp[valid,j]; gc=g[valid]
        y=d['labels'][rows,j]; count=np.bincount(ids,minlength=len(rows)); use=(y>=0)&(count>=q['min_views'])
        if len(np.unique(y[use]))!=2 or len(np.unique(d['region'][rows][use]))<q['min_regions']:
            raise ValueError(f'{c}: sampled support insufficient; increase sample_per_region')
        rho=float(model.rho[j]); best={}; curve=[]
        for g50 in q['g50_grid']:
            for beta in q['beta_grid']:
                agg=aggregate(ids,pc,gc,len(rows),rho,g50,beta,model.p_clip)
                for method in METHODS:
                    alphas=[1.] if method in ('mean','max','pi_mean') else q['alpha_grid']
                    for alpha in alphas:
                        s=probability(agg,method,alpha,rho)
                        loss=nll(y[use],s[use],w[use])
                        record=dict(nll=loss,g50=g50,beta=beta,alpha=alpha)
                        if method not in best or loss<best[method]['nll']-1e-12: best[method]=record
                        if method=='censored': curve.append(record)
        for method, record in best.items():
            keys=('g50','beta') if method=='pi_mean' else ('g50','beta','alpha') if method in ('pi_llr','censored') else ('alpha',) if method=='llr' else ()
            record['boundary_parameters']=[key for key in keys if record[key] in (min(q[key+'_grid']),max(q[key+'_grid']))]
        
        region_support = {}
        
        for region in np.unique(d["region"][rows]):
            region_mask = d["region"][rows] == region

            region_support[str(region)] = {
                "positive_sampled": int(
                    np.sum(region_mask & (y == 1))
                ),
                "positive_usable": int(
                    np.sum(region_mask & use & (y == 1))
                ),
                "negative_usable": int(
                    np.sum(region_mask & use & (y == 0))
                ),
            }

        profiles = {}

        for parameter in ("g50", "beta", "alpha"):
            profiles[parameter] = []

            for value in q[parameter + "_grid"]:
                profile_nll = min(
                    trial["nll"]
                    for trial in curve
                    if trial[parameter] == value
                )

                profiles[parameter].append({
                    "value": value,
                    "nll": profile_nll,
                    "delta_from_best": (
                        profile_nll - best["censored"]["nll"]
                    ),
                })

        sensitivity = None
        reference_grid = q.get("sensitivity_reference_grid")

        if reference_grid is not None:
            reference_trials = [
                trial
                for trial in curve
                if all(
                    trial[parameter]
                    in reference_grid[parameter + "_grid"]
                    for parameter in ("g50", "beta", "alpha")
                )
            ]

            if not reference_trials:
                raise ValueError(
                    "La griglia di riferimento non è contenuta "
                    "nella ricerca eseguita."
                )

            reference_best = min(
                reference_trials,
                key=lambda trial: trial["nll"],
            )

            sensitivity = {
                "reference_best": reference_best,
                "expanded_best": best["censored"],
                "nll_improvement": (
                    reference_best["nll"]
                    - best["censored"]["nll"]
                ),
            }
        
        result[c]=dict(
                rho=rho,
                methods=best,
                search=curve,
                sample_positive=int((y[use]==1).sum()),
                sample_negative=int((y[use]==0).sum()),
                region_support=region_support,
                positive_regions=np.unique(
                    d["region"][rows][use & (y == 1)]
                ).tolist(),
                profiles=profiles,
                sensitivity=sensitivity,
                supported_regions=np.unique(d['region'][rows][use]).tolist(),
                warnings=['Check boundary optima and flat losses; parameters are not physical measurements'])
    write_json(a.out,dict(protocol='supervised_a_v1',kind='fit',provenance=prov,config=q,
        base_params=p,results=result,fit_source_index=d['source_index'][rows].tolist(),
        fit_weighting=(
            "region x crack GT stratification; all crack positives; "
            "area divided by inclusion probability"
        )))

def check_artifact(artifact,prov,q,p):
    if artifact['protocol']!='supervised_a_v1' or artifact['provenance']!=prov or artifact['config']!=q or artifact['base_params']!=p:
        raise ValueError('Artifact/config/base parameters/provenance mismatch')

def score_split(a,d,model,q,artifact,split):
    rows=np.flatnonzero(d['split']==split)
    if not len(rows): raise ValueError('Empty '+split)
    n=len(rows); sums={c:{m:np.zeros(n) for m in METHODS} for c in q['classes']}
    counts={c:np.zeros(n,np.int64) for c in q['classes']}
    for obs in iter_observations(a.run,d['source_index'][rows]):
        g=model.view_gsd(obs['gsd_frontal'],obs['cos_theta'],obs['map_scale']); pp=probs_of(obs)
        ids=np.searchsorted(d['source_index'][rows],obs['source_index']); unique,inv=np.unique(ids,return_inverse=True)
        for c in q['classes']:
            j=list(DAMAGE_CLASSES).index(c); valid=np.isfinite(pp[:,j]); local=inv[valid]
            rho=artifact['results'][c]['rho']; raw=pp[valid,j]; gg=g[valid]
            first=True
            for method in METHODS:
                r=artifact['results'][c]['methods'][method]
                agg=aggregate(local,raw,gg,len(unique),rho,r['g50'],r['beta'],model.p_clip)
                if first:
                    counts[c][unique]+=agg['count']; first=False
                if method=='max':
                    sums[c][method][unique]=np.maximum(sums[c][method][unique],agg['max'])
                elif method=='mean': sums[c][method][unique]+=agg['mean']*agg['count']
                elif method=='pi_mean':
                    # iter_observations guarantees all views of each point in one block.
                    sums[c][method][unique]=agg[method]
                else: sums[c][method][unique]+=agg[method]
    scores={}
    for c in q['classes']:
        scores[c]={}
        for m in METHODS:
            x=sums[c][m]
            if m=='mean': x=x/np.maximum(counts[c],1)
            elif m not in ('max','pi_mean'):
                r=artifact['results'][c]['methods'][m]; x=sigmoid(logit(artifact['results'][c]['rho'])+r['alpha']*x)
            scores[c][m]=x
    return rows,counts,scores

def freeze(a):
    q,d,p,model,prov=context(a); artifact=read_json(a.fit); check_artifact(artifact,prov,q,p)
    rows,counts,scores=score_split(a,d,model,q,artifact,'C_cal'); thresholds={}; support={}
    for c in q['classes']:
        j=list(DAMAGE_CLASSES).index(c); y=d['labels'][rows,j]; use=(y>=0)&(counts[c]>=q['min_views']); w=d['area'][rows][use]
        if len(np.unique(y[use]))!=2 or len(np.unique(d['region'][rows][use]))<q['min_regions']: raise ValueError(c+': C_cal support')
        thresholds[c]={}; support[c]=dict(n=int(use.sum()))
        for m in METHODS:
            s=scores[c][m][use]
            values=[metrics(y[use],s,w,t)['f1'] for t in q['threshold_grid']]
            thresholds[c][m]=q['threshold_grid'][int(np.argmax(values))]
    result=copy.deepcopy(artifact); result.update(kind='frozen',fit_sha256=sha256_file(a.fit),thresholds=thresholds,
        threshold_split='C_cal',threshold_support=support,tie_policy='first threshold in config order')
    write_json(a.out,result)

def evaluate(a):
    q,d,p,model,prov=context(a); artifact=read_json(a.frozen); check_artifact(artifact,prov,q,p)
    if artifact['kind']!='frozen': raise ValueError('Frozen artifact required')
    rows,counts,scores=score_split(a,d,model,q,artifact,'T'); results={}
    for c in q['classes']:
        j=list(DAMAGE_CLASSES).index(c); y=d['labels'][rows,j]; use=(y>=0)&(counts[c]>=q['min_views']); known=y>=0
        if not use.any(): raise ValueError(c+': no T support')
        per={}
        for m in METHODS:
            per[m]=metrics(y[use],scores[c][m][use],d['area'][rows][use],artifact['thresholds'][c][m])
        reg={}
        for r in np.unique(d['region'][rows][use]):
            mask=use&(d['region'][rows]==r)
            reg[str(r)]={m:metrics(y[mask],scores[c][m][mask],d['area'][rows][mask],artifact['thresholds'][c][m]) for m in METHODS}
        results[c]=dict(methods=per,regions=reg,known_points=int(known.sum()),supported_points=int(use.sum()),
            area_coverage=float(d['area'][rows][use].sum()/d['area'][rows][known].sum()))
    write_json(a.out,dict(protocol='supervised_a_v1',frozen_sha256=sha256_file(a.frozen),provenance=prov,
        split='T',results=results,limitations=['No paired bootstrap','No axis/tolerance crack metrics','No cross-bridge test']))

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name,fun in [('preflight',preflight),('fit',fit),('freeze',freeze),('evaluate',evaluate)]:
        s=sub.add_parser(name); s.set_defaults(fun=fun)
        for key in ('run','regions','params','config','out'): s.add_argument('--'+key,required=True,type=Path)
        s.add_argument('--buffer',required=True,type=float)
        if name=='fit': s.add_argument('--preflight',required=True,type=Path)
        if name=='freeze': s.add_argument('--fit',required=True,type=Path)
        if name=='evaluate': s.add_argument('--frozen',required=True,type=Path)
    a=parser.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    a.fun(a)

if __name__=='__main__': main()
