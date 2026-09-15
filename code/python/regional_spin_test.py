"""Surface-based spin null for the regional Tini correlation.

Uses CRL 30-week parcellation labels on the topology-matched CHN 30-week
spheres.  Parcel centroids are rotated on the sphere and reassigned with a
bijective minimum-cost (Hungarian) match within each hemisphere.
"""

from __future__ import annotations

import base64
import json
import os
import re
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAP_KIND = os.environ.get("CORTICAL_SPIN_MAP", "fdrad").lower()
OUT = ROOT / "results" / "generated" / ("regional_radiality_spin_test" if MAP_KIND == "radiality" else "regional_spin_test")
OUT.mkdir(parents=True, exist_ok=True)
N_SPIN = 10000
SEED = 20260826


def norm_label(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_gifti_arrays(path: Path) -> tuple[list[np.ndarray], dict[int, str]]:
    root = ET.parse(path).getroot()
    labels = {}
    label_table = root.find("LabelTable")
    if label_table is not None:
        for item in label_table.findall("Label"):
            labels[int(item.attrib["Key"])] = item.text or ""
    arrays = []
    dtypes = {"NIFTI_TYPE_INT32": np.dtype("<i4"), "NIFTI_TYPE_FLOAT32": np.dtype("<f4")}
    for da in root.findall("DataArray"):
        dtype = dtypes[da.attrib["DataType"]]
        shape = tuple(int(da.attrib[f"Dim{i}"]) for i in range(int(da.attrib["Dimensionality"])))
        raw = base64.b64decode("".join((da.findtext("Data") or "").split()))
        if da.attrib["Encoding"] == "GZipBase64Binary":
            raw = zlib.decompress(raw)
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape, order="C")
        arrays.append(arr)
    return arrays, labels


def parcel_centroids(label_path: Path, sphere_path: Path) -> dict[str, np.ndarray]:
    label_arrays, table = read_gifti_arrays(label_path)
    sphere_arrays, _ = read_gifti_arrays(sphere_path)
    lab = label_arrays[0].astype(int)
    xyz = sphere_arrays[0].astype(float)
    xyz -= xyz.mean(axis=0)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    result = {}
    for key in np.unique(lab):
        if key == 0 or key not in table:
            continue
        v = xyz[lab == key].mean(axis=0)
        if np.linalg.norm(v) > 0:
            result[norm_label(table[key])] = v / np.linalg.norm(v)
    return result


def hungarian(cost: np.ndarray) -> np.ndarray:
    """Return column assignment for each row of a square cost matrix."""
    n = cost.shape[0]
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    ans = np.empty(n, dtype=int)
    for j in range(1, n + 1): ans[p[j] - 1] = j - 1
    return ans


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


left = parcel_centroids(
    ROOT / "data/atlases/crl_weekly_labels/30W_CRL_left.label.gii",
    ROOT / "data/atlases/chn_weekly_surface/30w/sub-Atlas_30w_ses-session1_left_sphere.surf.gii")
right = parcel_centroids(
    ROOT / "data/atlases/crl_weekly_labels/30W_CRL_right.label.gii",
    ROOT / "data/atlases/chn_weekly_surface/30w/sub-Atlas_30w_ses-session1_right_sphere.surf.gii")

if MAP_KIND == "radiality":
    source = pd.read_csv(ROOT / "results/generated/regional_radiality_gompertz/radiality_curvature_tini.csv")
    source = source.loc[source["joint_primary_pass"].astype(bool)].copy()
    source = source.rename(columns={"Tini_radiality": "Tinit-FDrad"})
else:
    source = pd.read_csv(ROOT / "data" / "source_data" / "fig4_regional_gompertz.csv")
    source = source.loc[source["Inclusion flag"].astype(bool)].copy()
source["key"] = source["Region Label"].map(norm_label)
source["hemi"] = source["Region Label"].str.extract(r"_([LR])$")[0]

missing = [k for k, h in zip(source["key"], source["hemi"]) if k not in (left if h == "L" else right)]
if missing:
    raise ValueError(f"Included regions missing from surface labels: {missing}")

coords = np.vstack([(left if h == "L" else right)[k] for k, h in zip(source["key"], source["hemi"])])
x = source["Tinit-FDrad"].to_numpy(float)
y = source["Tini_curvature"].to_numpy(float)
observed = float(np.corrcoef(x, y)[0, 1])

idx_l = np.where(source["hemi"].to_numpy() == "L")[0]
idx_r = np.where(source["hemi"].to_numpy() == "R")[0]

# Infer the coordinate reflection that best aligns homologous L/R centroids.
common_bases = sorted(set(k[:-1] for k in left if k.endswith("l")) & set(k[:-1] for k in right if k.endswith("r")))
candidate_reflections = [np.diag(s) for s in ((-1,1,1),(1,-1,1),(1,1,-1))]
reflection_scores = []
for M in candidate_reflections:
    reflection_scores.append(np.mean([np.linalg.norm(left[b+"l"] @ M - right[b+"r"]) for b in common_bases]))
reflection = candidate_reflections[int(np.argmin(reflection_scores))]

rng = np.random.default_rng(SEED)
null_coupled = np.empty(N_SPIN)
null_independent = np.empty(N_SPIN)
assignment_cost = np.empty((N_SPIN, 2))

def permutation_for(indices: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, float]:
    c = coords[indices]
    rotated = c @ rotation.T
    cost = np.sqrt(np.maximum(2 - 2 * np.clip(rotated @ c.T, -1, 1), 0))
    assignment = hungarian(cost)
    # At target parcel assignment[row], place the value originating at row.
    perm = np.empty(len(indices), dtype=int)
    perm[assignment] = np.arange(len(indices))
    return perm, float(cost[np.arange(len(indices)), assignment].mean())

for b in range(N_SPIN):
    R = random_rotation(rng)
    pl, cl = permutation_for(idx_l, R)
    # Mirrored/coupled rotation preserves homologous bilateral organization.
    Rr = reflection @ R @ reflection
    pr, cr = permutation_for(idx_r, Rr)
    xp = np.empty_like(x); xp[idx_l] = x[idx_l][pl]; xp[idx_r] = x[idx_r][pr]
    null_coupled[b] = np.corrcoef(xp, y)[0, 1]
    assignment_cost[b] = (cl, cr)

    Ri = random_rotation(rng)
    pri, _ = permutation_for(idx_r, Ri)
    xpi = np.empty_like(x); xpi[idx_l] = x[idx_l][pl]; xpi[idx_r] = x[idx_r][pri]
    null_independent[b] = np.corrcoef(xpi, y)[0, 1]

def p_two_sided(null: np.ndarray, obs: float) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))

summary = {
    "n_regions": int(len(source)), "n_left": int(len(idx_l)), "n_right": int(len(idx_r)),
    "map_kind": MAP_KIND,
    "observed_pearson_r": observed, "n_spins": N_SPIN, "seed": SEED,
    "coupled_mirrored_spin_p_two_sided": p_two_sided(null_coupled, observed),
    "independent_hemisphere_spin_p_two_sided_sensitivity": p_two_sided(null_independent, observed),
    "coupled_null_quantiles_2_5_50_97_5": np.quantile(null_coupled, [0.025, 0.5, 0.975]).tolist(),
    "reflection_axis_index": int(np.argmin(reflection_scores)),
    "reflection_alignment_scores": reflection_scores,
    "mean_chord_assignment_cost_left_right": assignment_cost.mean(axis=0).tolist(),
}
pd.DataFrame({"spin": np.arange(1, N_SPIN+1), "r_coupled": null_coupled,
              "r_independent": null_independent}).to_csv(OUT / "spin_null_distributions.csv", index=False)
source[["Region Label","Tinit-FDrad","Tini_curvature","hemi"]].to_csv(OUT / "included_regions.csv", index=False)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
