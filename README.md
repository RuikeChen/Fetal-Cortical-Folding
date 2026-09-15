# Cortical folding reproducibility package

This repository contains figure source data, de-identified analysis-level data,
statistical code, atlas access instructions, reference results, and scripts used
to run and plot the COMSOL simulations in the study.

## Directory structure

```text
data/
  source_data/   One CSV file for each figure-level source table
  derived/       De-identified participant-, session-, and ROI-level tables
  atlases/       Atlas citations, links, and expected local directory layout
code/
  python/        Statistical analyses, spatial tests, and figure generation
  matlab/        Whole-cortex and regional Gompertz analyses
  simulation/    COMSOL parameter-sweep and result-plotting scripts
results/
  reference/     Results used to check a new run
  generated/     Results created by the scripts
docs/            Data dictionary, analysis index, and dataset summary
```

## Data scope

The ZJU analysis tables contain study-specific participant and session IDs only.
Names, clinical identifiers, acquisition dates, and the private ID crosswalk are
not included. Raw human MRI data are not part of this package and require a
controlled-access process consistent with participant consent and institutional
approval.

The released code starts from de-identified analysis-level tables. Raw MRI,
intermediate images, and participant-specific cortical surfaces are not included.
The image-processing methods are described in the manuscript. Cortical surface
normals were generated with the Connectome Workbench `-surface-normals` command.

The public analysis tables contain 93 ZJU dMRI scans from 92 participants,
91 structural MRI scans from 89 participants, and 52 matched multimodal sessions
from 51 participants. Among participants with analyzable imaging, 41 contributed
dMRI only and 38 contributed structural MRI only. Two participants had repeated
structural MRI and one participant had repeated dMRI. See
`docs/dataset_summary.json` before using these counts in a manuscript.

## Python environment

Create the environment defined in `environment.yml`, activate it, and run scripts
from the repository root. Each script resolves input and output paths relative to
the repository.

Recommended order:

```text
python code/python/audit_regional_gompertz.py
python code/python/analyze_normalized_radiality.py
python code/python/fit_regional_radiality_gompertz.py
python code/python/regional_spin_test.py
python code/python/regional_robustness.py
python code/python/weekly_spin_test.py
python code/python/plot_weekly_spin_results.py
python code/python/bootstrap_tini_participant.py
python code/python/fit_longitudinal_models.py --shell b1000
python code/python/fit_longitudinal_models.py --shell b400
python code/python/longitudinal_permutation_tests.py
```

Set `CORTICAL_SPIN_MAP=radiality` for the regional spin script and
`CORTICAL_ROBUSTNESS_MAP=radiality` for the regional robustness script when
running the normalized-radiality sensitivity analysis.

## MATLAB analyses

The MATLAB scripts were prepared for MATLAB R2021a and require the Statistics
and Machine Learning Toolbox. Run:

```text
code/matlab/fit_whole_cortex_gompertz.m
code/matlab/fit_regional_gompertz.m
```

## COMSOL simulations

The simulation directory contains the scripts confirmed as part of the final
analysis:

- `Run_Tini.m` loads `Model_Tini.mph`.
- `Run_fiber_density.m` loads `Model_main.mph`.
- `Run_fiber_dispersion.m` loads `Model_main.mph`.

The two COMSOL model files are not included in this initial browser-uploaded
repository because each file exceeds GitHub's 25 MiB browser-upload limit.
They will be deposited separately or added later using Git Large File Storage:

- `Model_Tini.mph` (approximately 207 MiB)
- `Model_main.mph` (approximately 207 MiB)

The models were prepared for COMSOL Multiphysics 6.1 with LiveLink for MATLAB
R2021a. Set `COMSOL_MLI_PATH` to the COMSOL Multiphysics `mli` directory before
running these scripts.

## Reference results

Files under `results/reference` are compact outputs from the manuscript analysis.
Large permutation and bootstrap null distributions are omitted because the fixed
random seeds allow them to be regenerated. Compare regenerated summaries with the
reference files before revising manuscript values.

## Software versions

- MATLAB R2021a
- COMSOL Multiphysics 6.1 with LiveLink for MATLAB
- MRtrix3 3.0_RC3-135-g2b8e7d0c-dirty (64-bit build dated 17 December 2024)
- FSL 6.0.3
- Connectome Workbench 1.5.0

## Citation and license

Add the final repository DOI and software license before public release.

Requests for controlled access to the raw ZJU MRI data may be directed to
danwu.bme@zju.edu.cn. Access remains subject to the final ethics and data-use
conditions stated in the manuscript.
