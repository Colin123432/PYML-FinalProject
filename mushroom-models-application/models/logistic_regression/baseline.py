from sklearn.linear_model import LogisticRegression
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="logreg_baseline",
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        notes="One-hot encoded logistic regression.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
