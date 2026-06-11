# Capital Bikeshare SVM Classification Experiment

This project completes the machine learning lab requirement: use Capital Bikeshare
2025 trip data to predict `member_casual` with an SVM classifier.

## Run

The existing `cvlab` conda environment on this machine already contains the
required packages:

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n cvlab python scripts/run_svm_experiment.py
```

For a fresh environment, create one from `environment.yml`.

## Outputs

- `outputs/tables/`: sampling, cleaning, metrics, and factor-importance records
- `outputs/figures/`: data analysis, model evaluation, and flow-chart figures
- `outputs/models/`: trained SVM pipeline
- `outputs/processed/`: cleaned stratified sample used by the experiment
- `reports/report_outline.md`: Chinese report outline, methods, findings, and
  suggested explanation points for the final report

