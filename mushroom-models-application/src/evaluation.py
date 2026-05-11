from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn_fp_fn_tp": confusion_matrix(y_true, y_pred).ravel().tolist(),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def cross_validate_model(pipeline, X, y, n_splits=5, random_state=42, repeats=1) -> dict[str, Any]:
    """Run stratified k-fold or repeated stratified k-fold validation."""
    if repeats and repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=repeats,
            random_state=random_state,
        )
        cv_strategy = "RepeatedStratifiedKFold"
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_strategy = "StratifiedKFold"

    scoring = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    out = {"cv_folds": n_splits, "cv_repeats": repeats, "cv_strategy": cv_strategy}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(values))
        out[f"cv_{metric}_std"] = float(np.std(values))
    return out


def leaderboard_from_metrics(metrics_dir) -> pd.DataFrame:
    import json
    from pathlib import Path

    rows = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False)
