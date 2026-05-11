# Validation and Feature-Count Sweep Additions

This version adds a repeatable sweep framework for testing whether the same model families remain stable under different validation settings and feature counts.

## New or updated files

- `src/model_factories.py` centralizes estimator creation for sweep experiments.
- `src/sweeps.py` builds and runs groups of `ExperimentConfig` objects.
- `run_validation_sweeps.py` runs split-ratio, k-fold, top-k, and engineered-feature sweeps.
- `run_all.py` now accepts `--include-validation-sweeps` to run the quick sweep after the original baseline suite.
- `models/*/split_cv_sweep.py` runs validation-ratio/k-fold sweeps for one model family.
- `models/*/topk_sweep.py` runs top-k feature-count sweeps for one model family.
- `src/experiment.py` now includes sweep-friendly output toggles and supports repeated stratified k-fold CV through `cv_repeats`.
- `src/evaluation.py` now supports both `StratifiedKFold` and `RepeatedStratifiedKFold`.

## Main commands

Run the original suite only:

```bash
python run_all.py
```

Run the original suite plus the quick validation sweeps:

```bash
python run_all.py --include-validation-sweeps
```

Run only validation sweeps:

```bash
python run_validation_sweeps.py --mode quick
python run_validation_sweeps.py --mode full
```

Run selected models only:

```bash
python run_validation_sweeps.py --mode quick --models logreg random_forest
```

Save all per-experiment artifacts during sweeps:

```bash
python run_validation_sweeps.py --mode quick --save-full-outputs
```

## Sweep outputs

The sweep runner writes these summary outputs:

- `outputs/reports/validation_split_cv_sweep.csv`
- `outputs/reports/topk_feature_sweep.csv`
- `outputs/reports/feature_engineering_comparison.csv`
- `outputs/figures/validation_split_cv_sweep_f1.png`
- `outputs/figures/topk_feature_sweep_accuracy.png`
- `outputs/figures/feature_engineering_comparison_f1.png`

By default, sweep runs save compact summary outputs only. Use `--save-full-outputs` if you want every sweep run to save predictions, confusion matrices, feature-importance plots, and fitted models.
