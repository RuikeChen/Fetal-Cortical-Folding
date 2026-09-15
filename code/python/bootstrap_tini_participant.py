"""Participant-cluster bootstrap for the whole-brain Tini difference.

Uses the released whole-cortex observations and anonymous participant IDs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generated" / "tini_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260825)
N_BOOT = 5000


def model(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1.0, -50, 50)
    return a + b * np.exp(-np.exp(q))


def jacobian(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1.0, -50, 50)
    eq = np.exp(q)
    e = np.exp(-eq)
    j = np.empty((len(x), 4), dtype=float)
    j[:, 0] = 1.0
    j[:, 1] = e
    j[:, 2] = b * e * eq * (x - tini) * k
    j[:, 3] = -b * k * e * eq
    return j


def fit_gompertz(x: np.ndarray, y: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    theta = initial.astype(float).copy()
    lam = 1e-3
    old_sse = float(np.sum((y - model(theta, x)) ** 2))
    converged = False
    for _ in range(800):
        pred = model(theta, x)
        residual = y - pred
        j = jacobian(theta, x)
        a = j.T @ j
        g = j.T @ residual
        scale = np.diag(np.diag(a) + 1e-10)
        try:
            step = np.linalg.solve(a + lam * scale, g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        candidate = theta + step
        new_sse = float(np.sum((y - model(candidate, x)) ** 2))
        if np.isfinite(new_sse) and new_sse < old_sse:
            theta = candidate
            improvement = old_sse - new_sse
            old_sse = new_sse
            lam = max(lam / 3, 1e-12)
            if improvement < 1e-12 * max(1.0, old_sse) or np.linalg.norm(step) < 1e-9:
                converged = True
                break
        else:
            lam = min(lam * 10, 1e18)
    j = jacobian(theta, x)
    dof = max(len(x) - 4, 1)
    mse = old_sse / dof
    try:
        covariance = mse * np.linalg.inv(j.T @ j)
        se = np.sqrt(np.maximum(np.diag(covariance), 0))
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
    return theta, se, old_sse, converged


curv_source = pd.read_csv(ROOT / "data" / "derived" / "zju_whole_cortex_curvature.csv")
afd_source = pd.read_csv(ROOT / "data" / "derived" / "zju_whole_cortex_fdrad.csv")
curv = curv_source.rename(columns={
    "participant_id": "participant", "ga_weeks": "ga", "curvature": "value"
})[["participant", "ga", "value"]]
afd = afd_source.rename(columns={
    "participant_id": "participant", "ga_weeks": "ga", "fdrad": "value"
})[["participant", "ga", "value"]]


def scenario_data(name: str) -> pd.DataFrame:
    if name == "primary_corrected_metadata":
        return curv.copy()
    raise ValueError(name)


def cluster_bootstrap(curv_data: pd.DataFrame, scenario: str) -> tuple[dict, pd.DataFrame]:
    participants = np.array(sorted(set(curv_data["participant"]) | set(afd["participant"])))
    c_groups = {p: g.index.to_numpy() for p, g in curv_data.groupby("participant")}
    a_groups = {p: g.index.to_numpy() for p, g in afd.groupby("participant")}

    init_c = np.array([0.1, 0.5, math.log(0.5), 25.0])
    init_a = np.array([0.5, -0.5, math.log(0.5), 25.0])
    full_c = fit_gompertz(curv_data["ga"].to_numpy(), curv_data["value"].to_numpy(), init_c)
    full_a = fit_gompertz(afd["ga"].to_numpy(), afd["value"].to_numpy(), init_a)
    init_c = full_c[0]
    init_a = full_a[0]

    records = []
    for b in range(N_BOOT):
        sampled = RNG.choice(participants, size=len(participants), replace=True)
        ci = np.concatenate([c_groups[p] for p in sampled if p in c_groups])
        ai = np.concatenate([a_groups[p] for p in sampled if p in a_groups])
        fc = fit_gompertz(curv_data.loc[ci, "ga"].to_numpy(), curv_data.loc[ci, "value"].to_numpy(), init_c)
        fa = fit_gompertz(afd.loc[ai, "ga"].to_numpy(), afd.loc[ai, "value"].to_numpy(), init_a)
        tc, ta = fc[0][3], fa[0][3]
        valid = np.isfinite(tc) and np.isfinite(ta) and 18 < tc < 45 and 18 < ta < 45
        records.append((b + 1, tc, ta, tc - ta, valid, fc[3], fa[3]))

    reps = pd.DataFrame(records, columns=["replicate", "tini_curvature", "tini_fdrad", "delta_weeks", "valid", "curv_converged", "afd_converged"])
    valid = reps.loc[reps["valid"], "delta_weeks"].to_numpy()
    ci = np.percentile(valid, [2.5, 50, 97.5])
    summary = {
        "scenario": scenario,
        "participant_clusters": int(len(participants)),
        "curvature_rows": int(len(curv_data)),
        "afd_rows": int(len(afd)),
        "full_tini_curvature": float(full_c[0][3]),
        "full_se_curvature": float(full_c[1][3]),
        "full_tini_fdrad": float(full_a[0][3]),
        "full_se_fdrad": float(full_a[1][3]),
        "full_delta_weeks": float(full_c[0][3] - full_a[0][3]),
        "valid_bootstrap_replicates": int(len(valid)),
        "invalid_bootstrap_replicates": int(N_BOOT - len(valid)),
        "bootstrap_delta_ci_2_5": float(ci[0]),
        "bootstrap_delta_median": float(ci[1]),
        "bootstrap_delta_ci_97_5": float(ci[2]),
        "bootstrap_probability_delta_gt_0": float(np.mean(valid > 0)),
        "bootstrap_probability_delta_gt_1_week": float(np.mean(valid > 1.0)),
    }
    return summary, reps


summaries = []
for scenario in ["primary_corrected_metadata"]:
    summary, reps = cluster_bootstrap(scenario_data(scenario), scenario)
    summaries.append(summary)
    reps.to_csv(OUT / f"bootstrap_replicates_{scenario}.csv", index=False)

pd.DataFrame(summaries).to_csv(OUT / "bootstrap_summary.csv", index=False)
with open(OUT / "bootstrap_summary.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)
print(pd.DataFrame(summaries).to_string(index=False))
