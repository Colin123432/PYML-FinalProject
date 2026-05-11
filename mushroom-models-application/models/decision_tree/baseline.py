from sklearn.tree import DecisionTreeClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="decision_tree_baseline",
        estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
        notes="Interpretable shallow tree.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
