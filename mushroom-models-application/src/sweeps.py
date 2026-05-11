from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import DEFAULT_DATA_PATH, FIGURES_DIR, REPORTS_DIR, RANDOM_STATE
from .experiment import ExperimentConfig, run_experiment
from .model_factories import MODEL_FACTORIES, MODEL_DISPLAY_NAMES


def _compact_ratio(test_size: float) -> str:
    train_pct = int(round((1 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    return f"{train_pct}_{test_pct}"


def _sweep_output_config(config: ExperimentConfig, save_full_outputs: bool) -> ExperimentConfig:
    """Turn off large artifacts for broad sweeps unless requested."""
    config.save_predictions = save_full_outputs
    config.save_confusion_matrix = save_full_outputs
    config.save_feature_importance = save_full_outputs
    config.save_model = save_full_outputs
    config.save_shap = False
    return config


@dataclass(frozen=True)
class SweepPlan:
    """Defines the dimensions for validation and feature-count sweeps."""

    model_names: tuple[str, ...] = ("logreg", "decision_tree", "random_forest", "xgboost")
    test_sizes: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    cv_folds: tuple[int, ...] = (3, 5, 10)
    cv_repeats: tuple[int, ...] = (1,)
    top_feature_counts: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 10, 15, None)
    include_engineered_comparison: bool = True
    random_state: int = RANDOM_STATE


QUICK_SWEEP_PLAN = SweepPlan(
    model_names=("logreg", "decision_tree", "random_forest"),
    test_sizes=(0.20, 0.30),
    cv_folds=(3, 5),
    cv_repeats=(1,),
    top_feature_counts=(1, 3, 5, 7, 8, None),
    include_engineered_comparison=True,
)


FULL_SWEEP_PLAN = SweepPlan()


def build_split_cv_configs(
    plan: SweepPlan,
    save_full_outputs: bool = False,
) -> list[ExperimentConfig]:
    """Create configs testing the same model types across split ratios and k-fold counts."""
    configs: list[ExperimentConfig] = []
    for model_name in plan.model_names:
        factory = MODEL_FACTORIES[model_name]
        for test_size in plan.test_sizes:
            ratio_name = _compact_ratio(test_size)
            for folds in plan.cv_folds:
                for repeats in plan.cv_repeats:
                    repeat_suffix = f"_rep{repeats}" if repeats > 1 else ""
                    config = ExperimentConfig(
                        experiment_name=f"{model_name}_split_{ratio_name}_cv{folds}{repeat_suffix}",
                        estimator=factory(random_state=plan.random_state),
                        test_size=test_size,
                        cv_folds=folds,
                        cv_repeats=repeats,
                        random_state=plan.random_state,
                        notes=(
                            f"Validation sweep: {MODEL_DISPLAY_NAMES.get(model_name, model_name)} "
                            f"with stratified {int((1-test_size)*100)}/{int(test_size*100)} "
                            f"holdout split and {folds}-fold stratified CV."
                        ),
                    )
                    configs.append(_sweep_output_config(config, save_full_outputs))
    return configs


def build_topk_configs(
    plan: SweepPlan,
    save_full_outputs: bool = False,
) -> list[ExperimentConfig]:
    """Create configs testing how many top-ranked original features each model needs."""
    configs: list[ExperimentConfig] = []
    for model_name in plan.model_names:
        factory = MODEL_FACTORIES[model_name]
        for top_k in plan.top_feature_counts:
            k_name = "all" if top_k is None else str(top_k)
            config = ExperimentConfig(
                experiment_name=f"{model_name}_top{k_name}_features",
                estimator=factory(random_state=plan.random_state),
                top_n_features=top_k,
                test_size=0.20,
                cv_folds=5,
                random_state=plan.random_state,
                notes=(
                    f"Top-k feature sweep: {MODEL_DISPLAY_NAMES.get(model_name, model_name)} "
                    f"using {'all original features' if top_k is None else f'the top {top_k} original features'} "
                    "ranked by Cramér's V."
                ),
            )
            configs.append(_sweep_output_config(config, save_full_outputs))
    return configs


def build_engineered_comparison_configs(
    plan: SweepPlan,
    save_full_outputs: bool = False,
) -> list[ExperimentConfig]:
    """Create on/off engineered-feature comparisons for the same model types."""
    if not plan.include_engineered_comparison:
        return []

    configs: list[ExperimentConfig] = []
    for model_name in plan.model_names:
        if model_name == "dummy":
            continue
        factory = MODEL_FACTORIES[model_name]
        for enabled in (False, True):
            config = ExperimentConfig(
                experiment_name=f"{model_name}_{'engineered' if enabled else 'no_engineering'}_comparison",
                estimator=factory(random_state=plan.random_state),
                feature_engineering=enabled,
                test_size=0.20,
                cv_folds=5,
                random_state=plan.random_state,
                notes=(
                    f"Feature-engineering comparison for {MODEL_DISPLAY_NAMES.get(model_name, model_name)}; "
                    f"engineered features {'enabled' if enabled else 'disabled'}."
                ),
            )
            configs.append(_sweep_output_config(config, save_full_outputs))
    return configs


def run_configs(configs: Iterable[ExperimentConfig], csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    results = []
    for i, config in enumerate(configs, start=1):
        print(f"[{i}] Running {config.experiment_name} ...")
        results.append(run_experiment(config, csv_path=csv_path))
    return pd.DataFrame(results)


def save_sweep_tables_and_figures(
    split_cv_df: pd.DataFrame | None = None,
    topk_df: pd.DataFrame | None = None,
    engineered_df: pd.DataFrame | None = None,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if split_cv_df is not None and not split_cv_df.empty:
        split_cv_df.to_csv(REPORTS_DIR / "validation_split_cv_sweep.csv", index=False)
        _plot_split_cv_results(split_cv_df, FIGURES_DIR / "validation_split_cv_sweep_f1.png")

    if topk_df is not None and not topk_df.empty:
        topk_df.to_csv(REPORTS_DIR / "topk_feature_sweep.csv", index=False)
        _plot_topk_results(topk_df, FIGURES_DIR / "topk_feature_sweep_accuracy.png")

    if engineered_df is not None and not engineered_df.empty:
        engineered_df.to_csv(REPORTS_DIR / "feature_engineering_comparison.csv", index=False)
        _plot_engineered_results(engineered_df, FIGURES_DIR / "feature_engineering_comparison_f1.png")


def _plot_split_cv_results(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(11, 6))
    for model_name, group in df.groupby("model_class"):
        summary = group.groupby("test_size", as_index=False)["cv_f1_mean"].mean()
        plt.plot(summary["test_size"], summary["cv_f1_mean"], marker="o", label=model_name)
    plt.title("Validation sweep: mean CV F1 by test split ratio")
    plt.xlabel("Holdout test size")
    plt.ylabel("Mean CV F1")
    plt.ylim(0.9, 1.005)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_topk_results(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["top_n_features"].notna()].copy()
    if plot_df.empty:
        return
    plot_df["top_n_features"] = plot_df["top_n_features"].astype(int)

    plt.figure(figsize=(11, 6))
    for model_name, group in plot_df.groupby("model_class"):
        group = group.sort_values("top_n_features")
        plt.plot(group["top_n_features"], group["accuracy"], marker="o", label=model_name)
    plt.title("Top-k feature sweep: holdout accuracy by number of input features")
    plt.xlabel("Number of top-ranked original features")
    plt.ylabel("Holdout accuracy")
    plt.ylim(0.95, 1.005)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_engineered_results(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    labels = []
    values = []
    for _, row in plot_df.sort_values(["model_class", "feature_engineering"]).iterrows():
        suffix = "eng" if row["feature_engineering"] else "base"
        labels.append(f"{row['model_class']}\n{suffix}")
        values.append(row["cv_f1_mean"])

    plt.figure(figsize=(11, 6))
    plt.bar(labels, values)
    plt.title("Feature-engineering comparison: mean CV F1")
    plt.xlabel("Model / feature set")
    plt.ylabel("Mean CV F1")
    plt.ylim(0.95, 1.005)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
