# Data dictionary

## Common identifiers

| Field | Definition |
|---|---|
| `participant_id` | De-identified study participant identifier. |
| `session_id` | Within-participant session order based on gestational age. |
| `scan_id` | Modality-specific analysis scan identifier. |
| `ga_weeks` | Gestational age in decimal weeks. |
| `region` | Cortical region label. |

The participant ID crosswalk is not included. Repeated sessions share a
`participant_id`. The two participants who originally had the same romanized name
were assigned different public IDs before session numbering.

## ZJU derived data

| File | Unit of observation | Main fields |
|---|---|---|
| `zju_participants.csv` | Participant | Counts of structural and diffusion sessions. |
| `zju_sessions.csv` | Participant-session | Gestational age and modality availability. |
| `zju_diffusion_roi_metrics.csv.gz` | Diffusion scan-region | Gzip-compressed CSV containing `fdrad`, `fdtotal`, and `radiality`. |
| `zju_curvature_roi_metrics.csv.gz` | Structural scan-region | Gzip-compressed CSV containing `curvature`. |
| `zju_matched_roi_metrics.csv.gz` | Matched session-region | Gzip-compressed CSV containing curvature and all diffusion metrics. |
| `zju_fdrad_roi_wide.csv` | Diffusion scan | FDrad values in 78 region columns. |
| `zju_fdtotal_roi_wide.csv` | Diffusion scan | Total FD values in 78 region columns. |
| `zju_radiality_roi_wide.csv` | Diffusion scan | Normalized radiality values in 78 region columns. |
| `zju_curvature_roi_wide.csv` | Structural scan | Curvature values in 78 region columns. |
| `zju_whole_cortex_fdrad.csv` | Diffusion scan | Whole-cortex FDrad. |
| `zju_whole_cortex_curvature.csv` | Structural scan | Whole-cortex curvature. |

`radiality` is calculated as FDrad divided by FDtotal for the same scan and ROI.
The source FDtotal file contains the sum across the three fixel directions used in
the study. Curvature and FD units follow the definitions in the Methods.

## dHCP data

`dhcp_fdrad_b400_roi_wide.csv` contains the b = 400 s/mm2 sensitivity data. The
dHCP participant and session labels are pseudonymous identifiers from the source
dataset and are retained to support session alignment.

## Figure source data

Files in `data/source_data` preserve the values and table layout of the original
figure workbook while separating each figure-level table into its own CSV. Files
with two data blocks retain their title and header rows so the corresponding
analysis scripts can reconstruct each block without relying on Excel sheet names.

## Imaging data and surfaces

Raw MRI, intermediate images, and participant-specific cortical surfaces are not
included. Cortical surface normals were generated with Connectome Workbench 1.5.0
using the `-surface-normals` command. The analysis tables contain the ROI-level
values required by the released statistical scripts.

## Missing values

Blank CSV fields represent unavailable or inapplicable values. They must not be
replaced with zero. Inclusion fields in the regional Gompertz table are explicit
analysis flags rather than missing-data indicators.
