from __future__ import annotations

from src.config import DEFAULT_DATA_PATH
from src.sweeps import SweepPlan, build_topk_configs, run_configs, save_sweep_tables_and_figures


def main():
    plan = SweepPlan(model_names=("xgboost",))
    df = run_configs(build_topk_configs(plan, save_full_outputs=False), csv_path=DEFAULT_DATA_PATH)
    save_sweep_tables_and_figures(topk_df=df)
    print(df[["experiment_name", "top_n_features", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
