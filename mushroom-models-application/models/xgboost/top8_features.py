from xgboost import XGBClassifier
from src.config import DEFAULT_DATA_PATH
from src.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="xgboost_top8",
        estimator=XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        ),
        top_n_features=8,
        save_shap=True,
        notes="XGBoost with the top 8 original features only.",
    )
    print(run_experiment(config, DEFAULT_DATA_PATH))
