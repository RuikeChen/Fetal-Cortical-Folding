"""Fit regional normalized-radiality Gompertz curves with the manuscript rule."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generated" / "regional_radiality_gompertz"
OUT.mkdir(parents=True, exist_ok=True)


def betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a+b, a+1, a-1
    c = 1.0; d = 1.0-qab*x/qap
    if abs(d) < 1e-300: d = 1e-300
    d = 1.0/d; h = d
    for m in range(1, 301):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=max(abs(d),1e-300)*np.sign(d); c=1+aa/c; c=max(abs(c),1e-300)*np.sign(c); d=1/d; h*=d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=max(abs(d),1e-300)*np.sign(d); c=1+aa/c; c=max(abs(c),1e-300)*np.sign(c); d=1/d; delta=d*c; h*=delta
        if abs(delta-1) < 3e-12: break
    return float(h)


def ibeta(x: float, a: float, b: float) -> float:
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return bt*betacf(a,b,x)/a
    return 1-bt*betacf(b,a,1-x)/b


def t_p_two(t: float, df: int) -> float:
    return ibeta(df/(df+t*t), df/2, 0.5)


def f_p_upper(f: float, d1: int, d2: int) -> float:
    return ibeta(d2/(d2+d1*f), d2/2, d1/2)


def bh(p: np.ndarray) -> np.ndarray:
    q=np.full(len(p),np.nan); ok=np.isfinite(p); pv=p[ok]; order=np.argsort(pv); ranked=pv[order]*len(pv)/np.arange(1,len(pv)+1); ranked=np.minimum.accumulate(ranked[::-1])[::-1]; tmp=np.empty(len(pv)); tmp[order]=np.minimum(ranked,1); q[ok]=tmp; return q


def isoutlier_median(y: np.ndarray) -> np.ndarray:
    med=np.median(y); smad=1.482602218505602*np.median(np.abs(y-med))
    return np.abs(y-med)>3*smad if smad>0 else np.zeros(len(y),bool)


def model(th: np.ndarray, x: np.ndarray) -> np.ndarray:
    a,b,lk,t=th; k=math.exp(float(np.clip(lk,-10,5))); q=np.clip(-k*(x-t)+1,-50,50)
    return a+b*np.exp(-np.exp(q))


def jac(th: np.ndarray, x: np.ndarray) -> np.ndarray:
    a,b,lk,t=th; k=math.exp(float(np.clip(lk,-10,5))); q=np.clip(-k*(x-t)+1,-50,50); eq=np.exp(q); e=np.exp(-eq)
    return np.column_stack([np.ones(len(x)),e,b*e*eq*(x-t)*k,-b*k*e*eq])


def one_fit(x: np.ndarray, y: np.ndarray, init: np.ndarray):
    th=init.copy(); lam=1e-3; sse=float(np.sum((y-model(th,x))**2)); converged=False
    for _ in range(1500):
        r=y-model(th,x); j=jac(th,x); a=j.T@j; g=j.T@r
        try: step=np.linalg.solve(a+lam*np.diag(np.diag(a)+1e-12),g)
        except np.linalg.LinAlgError: lam*=10; continue
        cand=th+step; ns=float(np.sum((y-model(cand,x))**2))
        if np.isfinite(ns) and ns<sse:
            improvement=sse-ns; th=cand; sse=ns; lam=max(lam/3,1e-12)
            if improvement<1e-12*max(1,sse) or np.linalg.norm(step)<1e-9: converged=True; break
        else: lam=min(lam*10,1e18)
    return th,sse,converged


def fit(x: np.ndarray, y: np.ndarray):
    starts=[]
    for k in (0.1,0.25,0.5,1.0):
        for t in (23,26,29,32,35): starts.append(np.array([np.max(y),np.min(y)-np.max(y),math.log(k),t],float))
    candidates=[one_fit(x,y,s) for s in starts]; th,sse,conv=min(candidates,key=lambda z:z[1]); j=jac(th,x); df=len(x)-4
    try: cov=(sse/df)*np.linalg.inv(j.T@j); se=np.sqrt(np.maximum(np.diag(cov),0))
    except np.linalg.LinAlgError: se=np.full(4,np.nan)
    sst=float(np.sum((y-y.mean())**2)); r2=1-sse/sst; f=(r2/3)/((1-r2)/df) if 0<r2<1 else np.nan
    return th,se,r2,f_p_upper(f,3,df) if np.isfinite(f) else np.nan,t_p_two(th[3]/se[3],df) if se[3]>0 else np.nan,conv


afd=pd.read_csv(ROOT/'data/derived/zju_fdrad_roi_wide.csv')
fd=pd.read_csv(ROOT/'data/derived/zju_fdtotal_roi_wide.csv')
ga=afd['ga_weeks'].to_numpy(float); regions=afd.columns[4:].tolist(); radiality=afd.iloc[:,4:].to_numpy(float)/fd.iloc[:,4:].to_numpy(float)
rows=[]
for i,region in enumerate(regions):
    y0=radiality[:,i]; keep=np.isfinite(y0)&~isoutlier_median(y0); x=ga[keep]; y=y0[keep]
    th,se,r2,pfit,ptini,conv=fit(x,y)
    rows.append({'Region Label':region,'n':len(x),'ga_min':x.min(),'ga_max':x.max(),'a':th[0],'amplitude':th[1],'k':math.exp(th[2]),'Tini_radiality':th[3],'SE_Tini_radiality':se[3],'CI_width_radiality':3.92*se[3],'R2_radiality':r2,'p_fit_radiality':pfit,'p_Tini_radiality':ptini,'converged':conv})
res=pd.DataFrame(rows); res['q_fit_radiality']=bh(res.p_fit_radiality.to_numpy()); res['q_Tini_radiality']=bh(res.p_Tini_radiality.to_numpy())
res['primary_radiality_pass']=(res.q_fit_radiality<.05)&(res.q_Tini_radiality<.05)&(res.Tini_radiality>res.ga_min)&(res.Tini_radiality<res.ga_max)
res['strict_radiality_pass']=res.primary_radiality_pass&res.converged&(res.amplitude<0)&np.isfinite(res.CI_width_radiality)&(res.CI_width_radiality<=8)

curv=pd.read_csv(ROOT/'data/source_data/fig4_regional_gompertz.csv')
merged=curv[['Region Label','Tini_curvature','Inclusion flag']].merge(res,on='Region Label',how='inner')
# Curvature passed its original modality-specific criteria iff it participated in
# the original joint inclusion; this is conservative for the new joint analysis.
merged['joint_primary_pass']=merged['Inclusion flag'].astype(bool)&merged.primary_radiality_pass
merged['joint_strict_pass']=merged['Inclusion flag'].astype(bool)&merged.strict_radiality_pass
merged['delta_weeks']=merged.Tini_curvature-merged.Tini_radiality
res.to_csv(OUT/'all_radiality_fits.csv',index=False); merged.to_csv(OUT/'radiality_curvature_tini.csv',index=False)

summary={}
for criterion in ('joint_primary_pass','joint_strict_pass'):
    d=merged[merged[criterion]]; summary[criterion]={'n_regions':len(d),'pearson_r':float(np.corrcoef(d.Tini_radiality,d.Tini_curvature)[0,1]) if len(d)>3 else None,'mean_delta_weeks':float(d.delta_weeks.mean()),'median_delta_weeks':float(d.delta_weeks.median()),'proportion_radiality_earlier':float(np.mean(d.delta_weeks>0))}
(OUT/'fit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
