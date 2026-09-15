"""Reanalyse longitudinal data, separating prospective and co-developmental terms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--shell", choices=("b1000", "b400"), default="b1000",
    help="Diffusion shell used for the FDrad predictors.",
)
args = parser.parse_args()
OUT = ROOT / "results" / "generated" / ("longitudinal_b1000" if args.shell == "b1000" else "longitudinal_b400")
OUT.mkdir(parents=True, exist_ok=True)


def nelder_mead(fun, x0, step=0.4, max_iter=1200, tol=1e-9):
    n = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float)
        x[i] += step
        simplex.append(x)
    simplex = np.array(simplex)
    vals = np.array([fun(x) for x in simplex])
    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol and np.std(vals) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = fun(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        elif fr < vals[0]:
            xe = centroid + 2 * (xr - centroid)
            fe = fun(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = fun(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                vals[1:] = [fun(x) for x in simplex[1:]]
    i = np.argmin(vals)
    return simplex[i], vals[i]


def projections(v, s, r):
    m = v.reshape(s, r)
    grand = np.full_like(m, m.mean())
    subject = m.mean(axis=1, keepdims=True) - m.mean()
    subject = np.repeat(subject, r, axis=1)
    roi = m.mean(axis=0, keepdims=True) - m.mean()
    roi = np.repeat(roi, s, axis=0)
    interaction = m - grand - subject - roi
    return grand.ravel(), subject.ravel(), roi.ravel(), interaction.ravel()


def fit_crossed_lmm(y, x, s, r):
    n, p = x.shape

    def vinv(v, theta):
        ve, vs, vr = np.exp(theta)
        pg, ps, pr, pi = projections(v, s, r)
        return (pg / (ve + r * vs + s * vr)
                + ps / (ve + r * vs)
                + pr / (ve + s * vr)
                + pi / ve)

    def objective(theta):
        ve, vs, vr = np.exp(theta)
        logdet_v = (math.log(ve + r * vs + s * vr)
                    + (s - 1) * math.log(ve + r * vs)
                    + (r - 1) * math.log(ve + s * vr)
                    + (s - 1) * (r - 1) * math.log(ve))
        vx = np.column_stack([vinv(x[:, j], theta) for j in range(p)])
        xtvx = x.T @ vx
        try:
            beta = np.linalg.solve(xtvx, x.T @ vinv(y, theta))
            sign, logdet_x = np.linalg.slogdet(xtvx)
            if sign <= 0:
                return 1e100
        except np.linalg.LinAlgError:
            return 1e100
        resid = y - x @ beta
        quad = float(resid @ vinv(resid, theta))
        if quad <= 0 or not np.isfinite(quad):
            return 1e100
        return logdet_v + logdet_x + quad

    initial_var = max(np.var(y), 1e-8)
    theta, obj = nelder_mead(objective, np.log([initial_var * 0.5, initial_var * 0.25, initial_var * 0.25]))
    ve, vs, vr = np.exp(theta)

    def vinv_final(v):
        pg, ps, pr, pi = projections(v, s, r)
        return (pg / (ve + r * vs + s * vr)
                + ps / (ve + r * vs)
                + pr / (ve + s * vr)
                + pi / ve)

    vx = np.column_stack([vinv_final(x[:, j]) for j in range(p)])
    xtvx = x.T @ vx
    cov_beta = np.linalg.inv(xtvx)
    beta = cov_beta @ (x.T @ vinv_final(y))
    se = np.sqrt(np.diag(cov_beta))
    z = beta / se
    pval = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])
    return beta, se, z, pval, {"residual": ve, "subject": vs, "roi": vr, "objective": obj}


source_file = ROOT / "data" / "source_data" / "fig5_dhcp_longitudinal.csv"
raw = pd.read_csv(source_file, header=None)
curv = raw.iloc[2:34].copy()
curv.columns = raw.iloc[1].tolist()
if args.shell == "b1000":
    fdrad = raw.iloc[37:69].copy()
    fdrad.columns = raw.iloc[36].tolist()
else:
    fdrad = pd.read_csv(ROOT / "data" / "derived" / "dhcp_fdrad_b400_roi_wide.csv")
    fdrad = fdrad.rename(columns={
        "participant_id": "subj",
        "session_id": "ses",
        "ga_weeks": "ga",
    })

id_cols = ["subj", "ses", "ga"]
candidate_regions = [c for c in curv.columns if c not in id_cols and pd.notna(c) and c in set(fdrad.columns)]
regions = []
for region in candidate_regions:
    curv[region] = pd.to_numeric(curv[region], errors="coerce")
    fdrad[region] = pd.to_numeric(fdrad[region], errors="coerce")
    # Whole_CP is blank in the curvature source block and is not one of the
    # 32 regional observations used in the manuscript LME.
    if np.isfinite(curv[region]).all() and np.isfinite(fdrad[region]).all():
        regions.append(region)
if len(regions) != 32:
    raise ValueError(f"Expected 32 complete shared cortical regions; found {len(regions)}")
for d in [curv, fdrad]:
    d["ga"] = pd.to_numeric(d["ga"])

if len(curv) != 32 or len(fdrad) != 32:
    raise ValueError(f"Expected 32 scans in each modality; got curvature={len(curv)}, FDrad={len(fdrad)}")
curv_keys = set(map(tuple, curv[id_cols].astype(str).to_numpy()))
fdrad_keys = set(map(tuple, fdrad[id_cols].astype(str).to_numpy()))
if curv_keys != fdrad_keys:
    raise ValueError(f"Session mismatch: curvature-only={curv_keys-fdrad_keys}; FDrad-only={fdrad_keys-curv_keys}")

records = []
subjects = sorted(curv["subj"].unique())
for si, subject in enumerate(subjects):
    c = curv[curv["subj"] == subject].sort_values("ga")
    a = fdrad[fdrad["subj"] == subject].sort_values("ga")
    assert len(c) == 2 and len(a) == 2
    assert np.allclose(c["ga"].to_numpy(float), a["ga"].to_numpy(float))
    ga1, ga2 = c["ga"].to_numpy(float)
    for ri, region in enumerate(regions):
        c1, c2 = float(c.iloc[0][region]), float(c.iloc[1][region])
        f1, f2 = float(a.iloc[0][region]), float(a.iloc[1][region])
        records.append({
            "subject": subject, "subject_index": si, "roi": region, "roi_index": ri,
            "ga1": ga1, "delta_ga": ga2 - ga1,
            "curvature1": c1, "curvature2": c2,
            "fdrad1": f1, "fdrad2": f2,
            "delta_fdrad": f2 - f1,
            "rate_fdrad": (f2 - f1) / (ga2 - ga1),
        })
long = pd.DataFrame(records).sort_values(["subject_index", "roi_index"]).reset_index(drop=True)
long.to_csv(OUT / "longitudinal_analysis_table.csv", index=False)


def standardized(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


model_specs = {
    "prospective_baseline_only": ["curvature1", "fdrad1", "ga1", "delta_ga"],
    "codevelopment_difference": ["curvature1", "fdrad1", "delta_fdrad", "ga1", "delta_ga"],
    "codevelopment_rate": ["curvature1", "fdrad1", "rate_fdrad", "ga1", "delta_ga"],
    "followup_fdrad_equivalent": ["curvature1", "fdrad1", "fdrad2", "ga1", "delta_ga"],
}

all_results = []
variance_results = []
y = standardized(long["curvature2"])
for model_name, predictors in model_specs.items():
    x = np.column_stack([np.ones(len(long))] + [standardized(long[p]) for p in predictors])
    names = ["Intercept"] + predictors
    beta, se, z, pval, variances = fit_crossed_lmm(y, x, len(subjects), len(regions))
    for name, b, s_err, stat, p in zip(names, beta, se, z, pval):
        all_results.append({"model": model_name, "term": name, "std_beta": b, "se": s_err,
                            "ci95_low": b - 1.96 * s_err, "ci95_high": b + 1.96 * s_err,
                            "z": stat, "p_normal_approx": p})
    variance_results.append({"model": model_name, **variances})

pd.DataFrame(all_results).to_csv(OUT / "longitudinal_model_coefficients.csv", index=False)
pd.DataFrame(variance_results).to_csv(OUT / "longitudinal_variance_components.csv", index=False)
with open(OUT / "model_specifications.json", "w", encoding="utf-8") as f:
    json.dump({"diffusion_shell": args.shell, "models": model_specs}, f, indent=2)
print(pd.DataFrame(all_results).to_string(index=False))
