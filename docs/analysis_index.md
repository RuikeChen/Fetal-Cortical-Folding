# Analysis index

| Manuscript result | Primary input | Code | Reference output |
|---|---|---|---|
| Whole-cortex Gompertz timing | Whole-cortex ZJU tables | `code/matlab/fit_whole_cortex_gompertz.m` | Regenerated on run |
| Regional Gompertz timing | ZJU ROI-wide tables | `code/matlab/fit_regional_gompertz.m` | `regional_fit_audit.csv` |
| Regional fit-quality sensitivity | ZJU ROI-wide tables | `code/python/audit_regional_gompertz.py` | `regional_qc_sensitivity_summary.csv` |
| Regional spatial correlation | Regional Gompertz source data and atlas surfaces | `code/python/regional_spin_test.py` | `regional_spin_summary.json` |
| Bilateral and hemisphere robustness | Regional Gompertz source data | `code/python/regional_robustness.py` | `regional_robustness_summary.json` |
| Weekly FDrad-curvature coupling | Weekly atlas source data and atlas surfaces | `code/python/weekly_spin_test.py` | `weekly_spin_results.csv` |
| Figure 4e | Weekly spin results | `code/python/plot_weekly_spin_results.py` | `weekly_spin_plot_data.csv` |
| Participant-cluster Tini bootstrap | Whole-cortex ZJU tables | `code/python/bootstrap_tini_participant.py` | `tini_bootstrap_summary.csv` |
| Normalized-radiality regional timing | ZJU FDrad and FDtotal tables | `code/python/fit_regional_radiality_gompertz.py` | `regional_radiality_fit_summary.json` |
| Normalized-radiality developmental and curvature associations | De-identified ZJU ROI tables | `code/python/analyze_normalized_radiality.py` | `normalized_radiality_summary.json` |
| Longitudinal mixed models | dHCP longitudinal source data | `code/python/fit_longitudinal_models.py` | `longitudinal_model_coefficients.csv` |
| b = 400 sensitivity model | dHCP curvature and b = 400 FDrad | `code/python/fit_longitudinal_models.py --shell b400` | `longitudinal_b400_model_coefficients.csv` |
| Longitudinal permutation tests | Generated longitudinal analysis table | `code/python/longitudinal_permutation_tests.py` | `longitudinal_permutation_summary.json` |
| Fiber-density simulation | `Model_main.mph` | `code/simulation/Run_fiber_density.m` | Figure source data |
| Fiber-dispersion simulation | `Model_main.mph` | `code/simulation/Run_fiber_dispersion.m` | Figure source data |
| Fiber-timing simulation | `Model_Tini.mph` | `code/simulation/Run_Tini.m` | Figure source data |

The plotting scripts under `code/simulation` are retained as analysis records.
Their input workbook references still need to be replaced by the released CSV
simulation outputs after a complete COMSOL rerun.
