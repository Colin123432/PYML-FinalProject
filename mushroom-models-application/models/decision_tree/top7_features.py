from sklearn.tree import DecisionTreeClassifier

from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="decision_tree_top7_features",
        estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
        top_n_features=7,
        notes=(
            "Decision Tree using only the top 7 original features by Cramér's V. "
            "Selected after top-k sweep as the smallest tested feature subset with perfect performance."
        ),
    )

    print(run_experiment(config, DEFAULT_DATA_PATH))