from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_class_distribution(df: pd.DataFrame, output_path: str | Path):
    counts = df["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.rename(index={"e": "edible", "p": "poisonous"}).plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_missing_values(df: pd.DataFrame, output_path: str | Path):
    missing_counts = df.replace("?", pd.NA).isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if missing_counts.empty:
        ax.text(0.5, 0.5, "No missing values after '?' handling", ha="center", va="center")
        ax.axis("off")
    else:
        missing_counts.plot(kind="bar", ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_top_significance(significance_df: pd.DataFrame, output_path: str | Path, top_n: int = 10):
    top = significance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["cramers_v"])
    ax.set_title(f"Top {top_n} Features by Cramér's V")
    ax.set_xlabel("Cramér's V")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, output_path: str | Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["edible", "poisonous"], cmap="Blues", ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_feature_importance(importance_df: pd.DataFrame, output_path: str | Path, title: str, top_n: int = 12):
    top = importance_df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Aggregated importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def save_leaderboard(leaderboard_df: pd.DataFrame, output_path: str | Path):
    if leaderboard_df.empty:
        return
    plot_df = leaderboard_df.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["experiment_name"], plot_df["f1"])
    ax.set_title("Experiment Comparison by F1")
    ax.set_xlabel("F1 score")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
