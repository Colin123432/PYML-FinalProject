from sklearn.ensemble import RandomForestClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="random_forest_baseline",
        estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        save_shap=True,
        notes="Strong tree ensemble baseline.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
