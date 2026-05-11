from sklearn.linear_model import LogisticRegression
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="logreg_engineered",
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        feature_engineering=True,
        notes="Adds interaction crosses and domain-inspired flags.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
