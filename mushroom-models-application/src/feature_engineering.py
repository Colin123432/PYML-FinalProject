from __future__ import annotations

import pandas as pd

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "odor" in X.columns:
        odor = X["odor"].fillna("missing")
        X["odor_is_none"] = odor.eq("n").astype(str)
        X["odor_is_high_risk"] = odor.isin(["c", "f", "m", "p", "s", "y"]).astype(str)
    if "bruises" in X.columns:
        bruises = X["bruises"].fillna("missing")
        X["bruises_yes"] = bruises.eq("t").astype(str)
    if "gill-size" in X.columns:
        gill_size = X["gill-size"].fillna("missing")
        X["gill_is_narrow"] = gill_size.eq("n").astype(str)
    if "stalk-root" in X.columns:
        stalk_root = X["stalk-root"].fillna("missing")
        X["stalk_root_missing"] = stalk_root.eq("missing").astype(str)

    if {"odor", "spore-print-color"}.issubset(X.columns):
        X["odor_x_spore"] = (
            X["odor"].fillna("missing") + "__" + X["spore-print-color"].fillna("missing")
        )
    if {"odor", "bruises"}.issubset(X.columns):
        X["odor_x_bruises"] = (
            X["odor"].fillna("missing") + "__" + X["bruises"].fillna("missing")
        )
    return X
