"""Analyze normalized radiality using de-identified ROI-level data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "generated" / "normalized_radiality"
OUT.mkdir(parents=True, exist_ok=True)


def normal_p(z: float) -> float:
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def gompertz(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1.0, -50, 50)
    return a + b * np.exp(-np.exp(q))


def gompertz_jacobian(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, log_k, tini = theta
    k = math.exp(float(np.clip(log_k, -10, 5)))
    q = np.clip(-k * (x - tini) + 1.0, -50, 50)
    eq, e = np.exp(q), np.exp(-np.exp(q))
    return np.column_stack([
        np.ones(len(x)), e, b * e * eq * (x - tini) * k, -b * k * e * eq,
    ])


def fit_gompertz(x: np.ndarray, y: np.ndarray) -> dict:
    starts = [
        np.array([np.max(y), np.min(y) - np.max(y), math.log(k), tini], float)
        for k in (0.1, 0.25, 0.5, 1.0)
        for tini in (23, 26, 29, 32, 35)
    ]
    candidates = []
    for initial in starts:
        theta, damping = initial.copy(), 1e-3
        sse = float(np.sum((y - gompertz(theta, x)) ** 2))
        converged = False
        for _ in range(1500):
            residual = y - gompertz(theta, x)
            jac = gompertz_jacobian(theta, x)
            normal = jac.T @ jac
            try:
                step = np.linalg.solve(
                    normal + damping * np.diag(np.diag(normal) + 1e-12),
                    jac.T @ residual,
                )
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            candidate = theta + step
            new_sse = float(np.sum((y - gompertz(candidate, x)) ** 2))
            if np.isfinite(new_sse) and new_sse < sse:
                improvement, theta, sse = sse - new_sse, candidate, new_sse
                damping = max(damping / 3, 1e-12)
                if improvement < 1e-12 * max(1.0, sse) or np.linalg.norm(step) < 1e-9:
                    converged = True
                    break
            else:
                damping = min(damping * 10, 1e18)
        candidates.append((theta, sse, converged))
    theta, sse, converged = min(candidates, key=lambda item: item[1])
    jac = gompertz_jacobian(theta, x)
    covariance = (sse / (len(x) - 4)) * np.linalg.pinv(jac.T @ jac)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_scans": int(len(x)),
        "parameters": {"a": float(theta[0]), "b": float(theta[1]), "k": float(math.exp(theta[2])), "Tini": float(theta[3])},
        "standard_errors": {"a": float(se[0]), "b": float(se[1]), "log_k": float(se[2]), "Tini": float(se[3])},
        "Tini_95_CI": [float(theta[3] - 1.96 * se[3]), float(theta[3] + 1.96 * se[3])],
        "r_squared": float(1 - sse / sst),
        "converged": bool(converged),
    }


def cluster_covariance(y: np.ndarray, x: np.ndarray, labels: np.ndarray, k_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(y)), x])
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    meat = np.zeros((design.shape[1], design.shape[1]))
    groups = np.unique(labels)
    for group in groups:
        selected = labels == group
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (len(groups) / (len(groups) - 1)) * ((len(y) - 1) / max(len(y) - k_total, 1))
    covariance = correction * inverse @ meat @ inverse
    return beta, covariance, residual


def two_way_cluster(y: np.ndarray, x: np.ndarray, participant: np.ndarray, region: np.ndarray, k_total: int) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(y)), x])
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta

    def meat(labels: np.ndarray) -> np.ndarray:
        result = np.zeros((design.shape[1], design.shape[1]))
        groups = np.unique(labels)
        for group in groups:
            selected = labels == group
            score = design[selected].T @ residual[selected]
            result += np.outer(score, score)
        correction = (len(groups) / (len(groups) - 1)) * ((len(y) - 1) / max(len(y) - k_total, 1))
        return correction * result

    pair = np.array([f"{a}|{b}" for a, b in zip(participant, region)])
    covariance = inverse @ (meat(participant) + meat(region) - meat(pair)) @ inverse
    return beta, np.sqrt(np.maximum(np.diag(covariance), 0))


diffusion = pd.read_csv(ROOT / "data" / "derived" / "zju_diffusion_roi_metrics.csv.gz")
matched = pd.read_csv(ROOT / "data" / "derived" / "zju_matched_roi_metrics.csv.gz")
metrics = ["fdrad", "fdtotal", "radiality"]

development = {}
for metric in metrics:
    frame = diffusion[["participant_id", "region", "ga_weeks", metric]].dropna().copy()
    frame["response"] = frame[metric] - frame.groupby("region")[metric].transform("mean")
    frame["ga_centered"] = frame["ga_weeks"] - frame["ga_weeks"].mean()
    beta, se = two_way_cluster(
        frame["response"].to_numpy(), frame[["ga_centered"]].to_numpy(),
        frame["participant_id"].to_numpy(), frame["region"].to_numpy(), 79,
    )
    development[metric] = {
        "n_observations": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "slope_per_week": float(beta[1]),
        "slope_se_twoway_cluster": float(se[1]),
        "slope_z": float(beta[1] / se[1]),
        "slope_p_normal": normal_p(beta[1] / se[1]),
    }

means = diffusion.groupby(["participant_id", "session_id", "scan_id", "ga_weeks"], as_index=False)[metrics].mean()
whole_cortex = {
    metric: fit_gompertz(means["ga_weeks"].to_numpy(float), means[metric].to_numpy(float))
    for metric in ("fdrad", "radiality")
}

associations = {}
for metric in metrics:
    frame = matched[["participant_id", "region", "ga_weeks", "curvature", metric]].dropna().copy()
    for column in ("curvature", metric, "ga_weeks"):
        frame[column] = (frame[column] - frame[column].mean()) / frame[column].std(ddof=0)
        frame[column] -= frame.groupby("region")[column].transform("mean")
    beta, se = two_way_cluster(
        frame["curvature"].to_numpy(), frame[[metric, "ga_weeks"]].to_numpy(),
        frame["participant_id"].to_numpy(), frame["region"].to_numpy(), 80,
    )
    associations[metric] = {
        "n_matched_sessions": int(frame[["participant_id", "ga_weeks"]].drop_duplicates().shape[0]),
        "n_matched_participants": int(frame["participant_id"].nunique()),
        "standardized_beta_metric": float(beta[1]),
        "twoway_cluster_se": float(se[1]),
        "twoway_z": float(beta[1] / se[1]),
        "twoway_p_normal": normal_p(beta[1] / se[1]),
        "standardized_beta_ga": float(beta[2]),
    }

summary = {
    "alignment": {
        "n_dmri_scans": int(means.shape[0]),
        "n_dmri_participants": int(means["participant_id"].nunique()),
        "n_regions": int(diffusion["region"].nunique()),
        "n_matched_sessions": int(matched[["participant_id", "ga_weeks"]].drop_duplicates().shape[0]),
        "n_matched_participants": int(matched["participant_id"].nunique()),
    },
    "roi_fixed_effect_development": development,
    "whole_cortex_gompertz": whole_cortex,
    "curvature_associations_roi_fe_ga_adjusted": associations,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
means.to_csv(OUT / "cortical_mean_metrics.csv", index=False)
print(json.dumps(summary, indent=2))
