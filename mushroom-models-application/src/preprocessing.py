from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )

def make_pipeline(feature_names: list[str], estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_names)),
            ("model", estimator),
        ]
    )
