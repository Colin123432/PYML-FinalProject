from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RANDOM_STATE
from .data_io import load_data, prepare_xy
from .evaluation import cross_validate_model, evaluate_predictions
from .explainability import aggregate_one_hot_importance, save_shap_summary
from .feature_engineering import add_engineered_features
from .plotting import save_confusion_matrix, save_feature_importance
from .preprocessing import make_pipeline
from .significance import rank_features, select_top_features


@dataclass
class ExperimentConfig:
    """Configuration/recipe for one mushroom-classification experiment.

    The original project used this object to vary the estimator, split size, k-fold count,
    feature engineering toggle, and top-k feature selection.  It now also has lightweight
    output toggles so large sweeps can save metrics without writing hundreds of model and
    prediction files.
    """

    experiment_name: str
    estimator: object
    test_size: float = 0.2
    random_state: int = RANDOM_STATE
    feature_engineering: bool = False
    top_n_features: int | None = None
    cv_folds: int = 5
    cv_repeats: int = 1
    stratify_split: bool = True
    save_shap: bool = False
    notes: str = ""

    # Sweep-friendly output controls. Defaults preserve the original behavior.
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_feature_importance: bool = True
    save_model: bool = True


def _positive_class_probability(pipeline, X_test):
    """Return P(poisonous) when the estimator exposes predict_proba."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        classes = list(pipeline.classes_)
        positive_idx = classes.index(1) if 1 in classes else 1
        return pipeline.predict_proba(X_test)[:, positive_idx]
    except Exception:
        return None


def run_experiment(config: ExperimentConfig, csv_path: str | Path | None = None) -> dict:
    df = load_data(csv_path)
    X, y = prepare_xy(df)

    if config.top_n_features:
        X, ranking = select_top_features(X, y, config.top_n_features)
    else:
        ranking = rank_features(X, y)

    if config.feature_engineering:
        X = add_engineered_features(X)

    feature_names = X.columns.tolist()
    pipeline = make_pipeline(feature_names, config.estimator)

    stratify_arg = y if config.stratify_split else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_arg,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = _positive_class_probability(pipeline, X_test)

    metrics = evaluate_predictions(y_test, y_pred, y_proba=y_proba)
    metrics.update(
        cross_validate_model(
            pipeline,
            X,
            y,
            n_splits=config.cv_folds,
            random_state=config.random_state,
            repeats=config.cv_repeats,
        )
    )
    metrics.update(
        {
            "experiment_name": config.experiment_name,
            "model_class": pipeline.named_steps["model"].__class__.__name__,
            "n_features_input": len(feature_names),
            "selected_features": feature_names,
            "feature_engineering": config.feature_engineering,
            "top_n_features": config.top_n_features,
            "test_size": config.test_size,
            "train_size": 1 - config.test_size,
            "random_state": config.random_state,
            "stratify_split": config.stratify_split,
            "cv_folds": config.cv_folds,
            "cv_repeats": config.cv_repeats,
            "notes": config.notes,
            "top_ranked_features": ranking.head(10)["feature"].tolist(),
        }
    )

    if config.save_predictions:
        prediction_df = X_test.copy()
        prediction_df["actual"] = y_test.values
        prediction_df["predicted"] = y_pred
        if y_proba is not None:
            prediction_df["predicted_proba_poisonous"] = y_proba
        prediction_df.to_csv(PREDICTIONS_DIR / f"{config.experiment_name}_predictions.csv", index=False)

    if config.save_confusion_matrix:
        save_confusion_matrix(
            y_test,
            y_pred,
            FIGURES_DIR / f"{config.experiment_name}_confusion_matrix.png",
            title=f"{config.experiment_name} confusion matrix",
        )

    if config.save_feature_importance:
        try:
            importance_df = aggregate_one_hot_importance(pipeline, feature_names)
            importance_df.to_csv(METRICS_DIR / f"{config.experiment_name}_feature_importance.csv", index=False)
            save_feature_importance(
                importance_df,
                FIGURES_DIR / f"{config.experiment_name}_feature_importance.png",
                title=f"{config.experiment_name} feature importance",
            )
        except Exception:
            pass

    if config.save_shap:
        try:
            save_shap_summary(
                pipeline,
                X_train,
                FIGURES_DIR / f"{config.experiment_name}_shap_summary.png",
            )
        except Exception:
            pass

    if config.save_model:
        joblib.dump(pipeline, MODELS_DIR / f"{config.experiment_name}.joblib")

    with open(METRICS_DIR / f"{config.experiment_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
