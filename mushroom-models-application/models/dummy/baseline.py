from sklearn.dummy import DummyClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="dummy_baseline",
        estimator=DummyClassifier(strategy="most_frequent"),
        notes="Sanity-check baseline.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
