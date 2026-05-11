from sklearn.ensemble import RandomForestClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="random_forest_top8",
        estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        top_n_features=8,
        save_shap=True,
        notes="Uses only the top 8 original features by Cramér's V.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
