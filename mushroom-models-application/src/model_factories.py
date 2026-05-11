from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def make_dummy(random_state: int = 42):
    return DummyClassifier(strategy="most_frequent")


def make_logistic_regression(random_state: int = 42):
    return LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)


def make_decision_tree(random_state: int = 42):
    return DecisionTreeClassifier(max_depth=6, random_state=random_state)


def make_random_forest(random_state: int = 42):
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)


def make_xgboost(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
    )


MODEL_FACTORIES = {
    "dummy": make_dummy,
    "logreg": make_logistic_regression,
    "decision_tree": make_decision_tree,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}


MODEL_DISPLAY_NAMES = {
    "dummy": "Dummy baseline",
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
