from sklearn.ensemble import RandomForestClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="random_forest_split_30",
        estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        test_size=0.3,
        notes="Split sensitivity check using a 70/30 train-test split.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
