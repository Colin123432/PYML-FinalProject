from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def aggregate_one_hot_importance(pipeline, input_features: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    cat_pipe = pre.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["encoder"]
    categories = encoder.categories_

    if hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        raw_importance = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    rows = []
    idx = 0
    for feature_name, cats in zip(input_features, categories):
        width = len(cats)
        rows.append(
            {
                "feature": feature_name,
                "importance": float(np.sum(np.abs(raw_importance[idx : idx + width]))),
            }
        )
        idx += width
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

def save_shap_summary(pipeline, X_train, output_path: str | Path, max_samples: int = 500):
    try:
        import shap
    except Exception as exc:
        warnings.warn(f"SHAP not available: {exc}")
        return

    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    transformed = pre.transform(X_train)
    if transformed.shape[0] > max_samples:
        transformed = transformed[:max_samples]
    feature_names = pre.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
    except Exception as exc:
        warnings.warn(f"Skipping SHAP summary: {exc}")
        return

    fig = plt.figure(figsize=(10, 6))
    values = shap_values
    if isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, 1]
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
