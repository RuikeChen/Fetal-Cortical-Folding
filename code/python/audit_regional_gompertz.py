"""Audit regional Gompertz fits without selecting on Tini != 0 p-values."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generated" / "regional_gompertz_audit"
OUT.mkdir(parents=True, exist_ok=True)


def model(theta, x):
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1, -50, 50)
    return a + b * np.exp(-np.exp(q))


def jacobian(theta, x):
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1, -50, 50)
    eq = np.exp(q)
    e = np.exp(-eq)
    return np.column_stack([
        np.ones(len(x)), e, b * e * eq * (x - tini) * k, -b * k * e * eq,
    ])


def fit(x, y, initial):
    theta = np.asarray(initial, float).copy()
    lam = 1e-3
    sse = float(np.sum((y - model(theta, x)) ** 2))
    converged = False
    for _ in range(1000):
        pred = model(theta, x)
        r = y - pred
        j = jacobian(theta, x)
        a = j.T @ j
        g = j.T @ r
        try:
            step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-10), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        candidate = theta + step
        new_sse = float(np.sum((y - model(candidate, x)) ** 2))
        if np.isfinite(new_sse) and new_sse < sse:
            improvement = sse - new_sse
            theta = candidate
            sse = new_sse
            lam = max(lam / 3, 1e-12)
            if improvement < 1e-12 * max(1, sse) or np.linalg.norm(step) < 1e-9:
                converged = True
                break
        else:
            lam = min(lam * 10, 1e18)
    pred = model(theta, x)
    j = jacobian(theta, x)
    mse = sse / max(len(x) - 4, 1)
    try:
        cov = mse * np.linalg.inv(j.T @ j)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return theta, se, r2, sse, converged


curv = pd.read_csv(ROOT / "data" / "derived" / "zju_curvature_roi_wide.csv")
afd = pd.read_csv(ROOT / "data" / "derived" / "zju_fdrad_roi_wide.csv")
manuscript_qc = pd.read_csv(ROOT / "data" / "source_data" / "fig4_regional_gompertz.csv")

curv_ga = curv["ga_weeks"].astype(float).to_numpy()
afd_ga = afd["ga_weeks"].astype(float).to_numpy()
region_names = list(afd.columns[4:])

rows = []
for i, region in enumerate(region_names):
    cy = curv.iloc[:, i + 4].astype(float).to_numpy()
    ay = afd.iloc[:, i + 4].astype(float).to_numpy()
    fc = fit(curv_ga, cy, [0.1, 0.2, math.log(0.5), 26])
    fa = fit(afd_ga, ay, [0.5, -0.5, math.log(0.5), 26])
    qc_row = manuscript_qc.iloc[i]
    rows.append({
        "region": region,
        "tini_curvature": fc[0][3],
        "se_tini_curvature": fc[1][3],
        "ci_width_curvature": 3.92 * fc[1][3],
        "amplitude_curvature": fc[0][1],
        "k_curvature": math.exp(float(np.clip(fc[0][2], -10, 5))),
        "r2_curvature": fc[2],
        "converged_curvature": fc[4],
        "tini_fdrad": fa[0][3],
        "se_tini_fdrad": fa[1][3],
        "ci_width_fdrad": 3.92 * fa[1][3],
        "amplitude_fdrad": fa[0][1],
        "k_fdrad": math.exp(float(np.clip(fa[0][2], -10, 5))),
        "r2_fdrad": fa[2],
        "converged_fdrad": fa[4],
        "delta_tini": fc[0][3] - fa[0][3],
        "manuscript_tini_curvature": qc_row["Tini_curvature"],
        "manuscript_tini_fdrad": qc_row["Tinit-FDrad"],
        "manuscript_inclusion": int(qc_row["Inclusion flag"]),
    })

result = pd.DataFrame(rows)
result["qc_basic"] = (
    result["converged_curvature"] & result["converged_fdrad"]
    & (result["amplitude_curvature"] > 0) & (result["amplitude_fdrad"] < 0)
    & result["tini_curvature"].between(22.5, 39.0)
    & result["tini_fdrad"].between(22.5, 39.0)
)
for width in [4, 6, 8, 10]:
    result[f"qc_ciwidth_{width}w"] = (
        result["qc_basic"]
        & (result["ci_width_curvature"] <= width)
        & (result["ci_width_fdrad"] <= width)
    )
for r2 in [0.1, 0.2, 0.3, 0.4]:
    result[f"qc_r2_{str(r2).replace('.', '_')}"] = (
        result["qc_basic"]
        & (result["r2_curvature"] >= r2)
        & (result["r2_fdrad"] >= r2)
    )

result.to_csv(OUT / "regional_fit_audit.csv", index=False)
summary = []
for col in ["manuscript_inclusion", "qc_basic", "qc_ciwidth_4w", "qc_ciwidth_6w", "qc_ciwidth_8w", "qc_ciwidth_10w", "qc_r2_0_1", "qc_r2_0_2", "qc_r2_0_3", "qc_r2_0_4"]:
    mask = result[col].astype(bool)
    if mask.sum() >= 4:
        r = np.corrcoef(result.loc[mask, "tini_curvature"], result.loc[mask, "tini_fdrad"])[0, 1]
        med_delta = result.loc[mask, "delta_tini"].median()
    else:
        r = np.nan
        med_delta = np.nan
    summary.append({"criterion": col, "n_regions": int(mask.sum()), "pearson_r": r, "median_delta_weeks": med_delta})
pd.DataFrame(summary).to_csv(OUT / "regional_qc_sensitivity_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
