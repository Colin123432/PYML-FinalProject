from sklearn.ensemble import RandomForestClassifier

from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment


if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="random_forest_top7_features",
        estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        top_n_features=7,
        save_shap=True,
        notes=(
            "Random Forest using only the top 7 original features by Cramér's V. "
            "Selected after top-k sweep as the smallest tested feature subset with perfect performance."
        ),
    )

    print(run_experiment(config, DEFAULT_DATA_PATH))