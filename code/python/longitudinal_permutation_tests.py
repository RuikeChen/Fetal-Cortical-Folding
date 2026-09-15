"""Participant-level permutation tests for the prospective FDrad model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated/longitudinal_permutation'; OUT.mkdir(parents=True,exist_ok=True)
N_PERM=100000; RNG=np.random.default_rng(20260826)
data=pd.read_csv(ROOT/'results/generated/longitudinal_b1000/longitudinal_analysis_table.csv').sort_values(['subject_index','roi_index']).reset_index(drop=True)
subjects=data.subject.unique(); regions=data.roi.unique(); ns,nr=len(subjects),len(regions)
assert len(data)==ns*nr and ns==16

def z(v):
    v=np.asarray(v,float); return (v-v.mean())/v.std(ddof=1)

def roi_demean(v):
    m=np.asarray(v,float).reshape(ns,nr); return (m-m.mean(axis=0,keepdims=True)).ravel()

def residual(v,Z):
    return v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]

def analyse(outcome_name,outcome,nuisance):
    y=roi_demean(z(outcome)); x=roi_demean(z(data.fdrad1))
    Z=np.column_stack([roi_demean(z(data[c])) for c in nuisance])
    # No intercept is needed after ROI demeaning.
    yr=residual(y,Z); xr=residual(x,Z); beta=float(xr@yr/(xr@xr)); corr=float(np.corrcoef(xr,yr)[0,1])
    xm=xr.reshape(ns,nr); ym=yr.reshape(ns,nr); null=np.empty(N_PERM)
    for b in range(N_PERM):
        xp=xm[RNG.permutation(ns)].ravel(); null[b]=float(xp@yr/(xp@xp))
    p=float((1+np.sum(np.abs(null)>=abs(beta)))/(N_PERM+1))
    loo=[]
    for i in range(ns):
        keep=np.ones(ns,bool); keep[i]=False; xx=xm[keep].ravel(); yy=ym[keep].ravel(); loo.append(float(xx@yy/(xx@xx)))
    pd.DataFrame({'permuted_beta':null}).to_csv(OUT/f'{outcome_name}_permutation_null.csv',index=False)
    return {'outcome':outcome_name,'n_subjects':ns,'n_regions':nr,'standardized_beta':beta,'partial_correlation':corr,'participant_profile_permutation_p_two_sided':p,'null_95_interval':np.quantile(null,[.025,.975]).tolist(),'leave_one_subject_out_beta_range':[min(loo),max(loo)],'nuisance_covariates':nuisance}

results=[]
# Standard ANCOVA: follow-up outcome adjusted for its baseline value.
results.append(analyse('followup_curvature_ancova',data.curvature2,['curvature1','ga1','delta_ga']))
# Secondary change-rate parameterization. Baseline curvature is retained to
# reduce regression-to-the-mean bias; this is not treated as an independent study.
rate=(data.curvature2-data.curvature1)/data.delta_ga
results.append(analyse('curvature_change_rate',rate,['curvature1','ga1','delta_ga']))
(OUT/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
