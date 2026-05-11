from __future__ import annotations

from src.config import DEFAULT_DATA_PATH
from src.sweeps import SweepPlan, build_split_cv_configs, run_configs, save_sweep_tables_and_figures


def main():
    plan = SweepPlan(model_names=("random_forest",))
    df = run_configs(build_split_cv_configs(plan, save_full_outputs=False), csv_path=DEFAULT_DATA_PATH)
    save_sweep_tables_and_figures(split_cv_df=df)
    print(df[["experiment_name", "test_size", "cv_folds", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
