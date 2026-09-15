"""Surface spin tests for weekly atlas FDrad-curvature correlations (23-38w)."""

from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generated" / "weekly_spin_test"
OUT.mkdir(parents=True, exist_ok=True)
N_SPIN = 10000
SEED = 20260831


def norm_label(s):
    s = str(s).replace(".left", "_L").replace(".right", "_R")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_gifti(path):
    root = ET.parse(path).getroot()
    table = {}
    lt = root.find("LabelTable")
    if lt is not None:
        for item in lt.findall("Label"):
            table[int(item.attrib["Key"])] = item.text or ""
    dtypes = {"NIFTI_TYPE_INT32": np.dtype("<i4"), "NIFTI_TYPE_FLOAT32": np.dtype("<f4")}
    arrays = []
    for da in root.findall("DataArray"):
        dtype = dtypes[da.attrib["DataType"]]
        shape = tuple(int(da.attrib[f"Dim{i}"]) for i in range(int(da.attrib["Dimensionality"])))
        raw = base64.b64decode("".join((da.findtext("Data") or "").split()))
        if da.attrib["Encoding"] == "GZipBase64Binary": raw = zlib.decompress(raw)
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(shape))
    return arrays, table


def centroids(label_path, sphere_path):
    la, table = read_gifti(label_path); sa, _ = read_gifti(sphere_path)
    lab = la[0].astype(int); xyz = sa[0].astype(float); xyz -= xyz.mean(0); xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    out = {}
    for key in np.unique(lab):
        if key and key in table:
            v = xyz[lab == key].mean(0)
            if np.linalg.norm(v): out[norm_label(table[key])] = v / np.linalg.norm(v)
    return out


def hungarian(cost):
    n = len(cost); u=np.zeros(n+1); v=np.zeros(n+1); p=np.zeros(n+1,int); way=np.zeros(n+1,int)
    for i in range(1,n+1):
        p[0]=i; j0=0; minv=np.full(n+1,np.inf); used=np.zeros(n+1,bool)
        while True:
            used[j0]=True; i0=p[j0]; delta=np.inf; j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=cost[i0-1,j-1]-u[i0]-v[j]
                    if cur<minv[j]: minv[j]=cur; way[j]=j0
                    if minv[j]<delta: delta=minv[j]; j1=j
            for j in range(n+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: minv[j]-=delta
            j0=j1
            if p[j0]==0: break
        while True:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
            if j0==0: break
    ans=np.empty(n,int)
    for j in range(1,n+1): ans[p[j]-1]=j-1
    return ans


def rotation(rng):
    q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def assignment(coords, R):
    rot=coords@R.T; cost=np.sqrt(np.maximum(2-2*np.clip(rot@coords.T,-1,1),0)); a=hungarian(cost)
    perm=np.empty(len(coords),int); perm[a]=np.arange(len(coords)); return perm


def corr_rows(x, y):
    xc=x-x.mean(axis=1,keepdims=True); yc=y-y.mean(axis=1,keepdims=True)
    return np.sum(xc*yc,axis=1)/np.sqrt(np.sum(xc*xc,axis=1)*np.sum(yc*yc,axis=1))


def bh(p):
    p=np.asarray(p); o=np.argsort(p); q0=p[o]*len(p)/np.arange(1,len(p)+1); q0=np.minimum.accumulate(q0[::-1])[::-1]
    q=np.empty(len(p)); q[o]=np.minimum(q0,1); return q


# Read the two 16x78 weekly blocks from the source-data sheet.
raw=pd.read_csv(ROOT/'data/source_data/fig4_weekly_atlas.csv',header=None)
header_curv=raw.iloc[1,1:79].tolist(); ga=raw.iloc[2:18,0].to_numpy(int); curv=raw.iloc[2:18,1:79].to_numpy(float)
header_fd=raw.iloc[20,1:79].tolist(); ga_fd=raw.iloc[21:37,0].to_numpy(int); fdrad=raw.iloc[21:37,1:79].to_numpy(float)
if not np.array_equal(ga,ga_fd) or [norm_label(x) for x in header_curv] != [norm_label(x) for x in header_fd]:
    raise ValueError('Weekly curvature and FDrad blocks are not aligned')
keys=[norm_label(x) for x in header_curv]

left=centroids(ROOT/'data/atlases/crl_weekly_labels/30W_CRL_left.label.gii',ROOT/'data/atlases/chn_weekly_surface/30w/sub-Atlas_30w_ses-session1_left_sphere.surf.gii')
right=centroids(ROOT/'data/atlases/crl_weekly_labels/30W_CRL_right.label.gii',ROOT/'data/atlases/chn_weekly_surface/30w/sub-Atlas_30w_ses-session1_right_sphere.surf.gii')
hemi=np.array(['L' if str(x).endswith('.left') else 'R' for x in header_curv]); il=np.where(hemi=='L')[0]; ir=np.where(hemi=='R')[0]
missing=[k for k,h in zip(keys,hemi) if k not in (left if h=='L' else right)]
if missing: raise ValueError(f'Missing surface labels: {missing}')
cl=np.vstack([left[keys[i]] for i in il]); cr=np.vstack([right[keys[i]] for i in ir])

# x-axis reflection best aligns homologous fetal spherical coordinates.
M=np.diag([-1,1,1]); rng=np.random.default_rng(SEED); perms=np.empty((N_SPIN,78),int); perms_ind=np.empty_like(perms)
for b in range(N_SPIN):
    R=rotation(rng); pl=assignment(cl,R); pr=assignment(cr,M@R@M); pri=assignment(cr,rotation(rng))
    perms[b,il]=il[pl]; perms[b,ir]=ir[pr]; perms_ind[b,il]=il[pl]; perms_ind[b,ir]=ir[pri]

obs=np.array([np.corrcoef(fdrad[i],curv[i])[0,1] for i in range(len(ga))])
null=np.empty((N_SPIN,len(ga))); null_ind=np.empty_like(null)
for i in range(len(ga)):
    yy=np.repeat(curv[i][None,:],N_SPIN,axis=0)
    null[:,i]=corr_rows(fdrad[i][perms],yy); null_ind[:,i]=corr_rows(fdrad[i][perms_ind],yy)
p=(1+np.sum(np.abs(null)>=np.abs(obs),axis=0))/(N_SPIN+1)
p_ind=(1+np.sum(np.abs(null_ind)>=np.abs(obs),axis=0))/(N_SPIN+1)
q=bh(p); max_abs=np.max(np.abs(null),axis=1); p_fwer=(1+np.sum(max_abs[:,None]>=np.abs(obs)[None,:],axis=0))/(N_SPIN+1)
result=pd.DataFrame({'GA':ga,'pearson_r':obs,'spin_p_two_sided':p,'spin_q_BH_16weeks':q,'spin_p_maxT_FWER':p_fwer,'independent_hemi_spin_p_sensitivity':p_ind})
result.to_csv(OUT/'weekly_spin_results.csv',index=False)
pd.DataFrame(null,columns=[f'{w}w' for w in ga]).to_csv(OUT/'weekly_coupled_spin_null.csv',index=False)
summary={'n_regions_per_week':78,'n_spins':N_SPIN,'seed':SEED,'significant_weeks_BH_q_lt_0_05':result.loc[result.spin_q_BH_16weeks<.05,'GA'].tolist(),'significant_weeks_maxT_FWER_lt_0_05':result.loc[result.spin_p_maxT_FWER<.05,'GA'].tolist()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(result.to_string(index=False)); print(json.dumps(summary,indent=2))
