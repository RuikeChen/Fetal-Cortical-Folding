"""Partial spatial robustness checks using bilateral anatomical pairs as clusters."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAP_KIND = os.environ.get("CORTICAL_ROBUSTNESS_MAP", "fdrad").lower()
OUT = ROOT / "results" / "generated" / ("regional_radiality_robustness" if MAP_KIND == "radiality" else "regional_spatial_robustness")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260825)
N = 20000

if MAP_KIND == "radiality":
    data = pd.read_csv(ROOT / "results/generated/regional_radiality_gompertz/radiality_curvature_tini.csv")
    data = data[data["joint_primary_pass"].astype(bool)].copy()
    data["Tinit-FDrad"] = data["Tini_radiality"]
else:
    data = pd.read_csv(ROOT / "data" / "source_data" / "fig4_regional_gompertz.csv")
    data = data[data["Inclusion flag"].astype(bool)].copy()
data["pair"] = data["Region Label"].map(lambda x: re.sub(r"_[LR]$", "", str(x)))
data["hemisphere"] = data["Region Label"].map(lambda x: str(x)[-1] if str(x).endswith(("_L", "_R")) else "U")
data["delta"] = data["Tini_curvature"] - data["Tinit-FDrad"]

observed_r = float(np.corrcoef(data["Tini_curvature"], data["Tinit-FDrad"])[0, 1])
observed_mean_delta = float(data["delta"].mean())
observed_median_delta = float(data["delta"].median())
observed_prop_fdrad_earlier = float(np.mean(data["delta"] > 0))

groups = {k: g.index.to_numpy() for k, g in data.groupby("pair")}
pair_names = np.array(sorted(groups))
boot_r, boot_mean, boot_median, boot_prop = [], [], [], []
for _ in range(N):
    sampled = RNG.choice(pair_names, size=len(pair_names), replace=True)
    idx = np.concatenate([groups[p] for p in sampled])
    b = data.loc[idx]
    boot_r.append(np.corrcoef(b["Tini_curvature"], b["Tinit-FDrad"])[0, 1])
    boot_mean.append(b["delta"].mean())
    boot_median.append(b["delta"].median())
    boot_prop.append(np.mean(b["delta"] > 0))

# Permute FDrad timing separately within hemispheres. This preserves hemisphere
# composition but is not a full cortical spatial null.
perm_r = []
for _ in range(N):
    perm = data["Tinit-FDrad"].copy()
    for hemi in data["hemisphere"].unique():
        idx = data.index[data["hemisphere"] == hemi]
        perm.loc[idx] = RNG.permutation(perm.loc[idx].to_numpy())
    perm_r.append(np.corrcoef(data["Tini_curvature"], perm)[0, 1])
perm_r = np.asarray(perm_r)
perm_p_two_sided = float((1 + np.sum(np.abs(perm_r) >= abs(observed_r))) / (N + 1))

summary = {
    "map_kind": MAP_KIND,
    "n_regions": int(len(data)),
    "n_bilateral_pair_clusters": int(len(pair_names)),
    "observed_pearson_r": observed_r,
    "bilateral_cluster_bootstrap_r_ci95": [float(x) for x in np.percentile(boot_r, [2.5, 97.5])],
    "hemisphere_stratified_permutation_p_two_sided": perm_p_two_sided,
    "observed_mean_delta_weeks": observed_mean_delta,
    "bilateral_cluster_bootstrap_mean_delta_ci95": [float(x) for x in np.percentile(boot_mean, [2.5, 97.5])],
    "observed_median_delta_weeks": observed_median_delta,
    "bilateral_cluster_bootstrap_median_delta_ci95": [float(x) for x in np.percentile(boot_median, [2.5, 97.5])],
    "observed_proportion_regions_fdrad_earlier": observed_prop_fdrad_earlier,
    "bilateral_cluster_bootstrap_proportion_ci95": [float(x) for x in np.percentile(boot_prop, [2.5, 97.5])],
    "limitation": "Bilateral clustering and hemisphere-stratified permutation do not preserve full cortical geodesic spatial autocorrelation; ROI centroids or a labelled surface mesh are required for a proper spatial null.",
}
with open(OUT / "regional_robustness_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
pd.DataFrame({"bootstrap_r": boot_r, "bootstrap_mean_delta": boot_mean, "bootstrap_median_delta": boot_median, "bootstrap_proportion_fdrad_earlier": boot_prop}).to_csv(OUT / "bilateral_cluster_bootstrap.csv", index=False)
pd.DataFrame({"permuted_r": perm_r}).to_csv(OUT / "hemisphere_stratified_permutation.csv", index=False)
print(json.dumps(summary, ensure_ascii=False, indent=2))
