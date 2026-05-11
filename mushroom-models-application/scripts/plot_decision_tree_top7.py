from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "decision_tree_top7_features.joblib"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "decision_tree_top7_features_tree.png"
TEXT_PATH = PROJECT_ROOT / "outputs" / "reports" / "decision_tree_top7_features_rules.txt"


def get_feature_names(pipeline):
    """
    Extract one-hot encoded feature names from the preprocessing step.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out()


def main():
    pipeline = joblib.load(MODEL_PATH)

    tree_model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    class_names = [str(c) for c in tree_model.classes_]

    # Large figure because one-hot encoded trees can be wide
    plt.figure(figsize=(32, 18))

    """plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
    )"""

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=9,
        max_depth=3,
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    # Also save text rules, which are often easier to read than the image
    rules = export_text(
        tree_model,
        feature_names=list(feature_names),
        spacing=3,
        decimals=3,
    )

    TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_PATH.write_text(rules)

    print(f"Saved tree diagram to: {FIGURE_PATH}")
    print(f"Saved tree rules to:   {TEXT_PATH}")


if __name__ == "__main__":
    main()