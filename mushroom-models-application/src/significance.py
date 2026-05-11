from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((k_corr - 1), (r_corr - 1))
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0

def rank_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    y_named = y.map({0: "e", 1: "p"})
    for col in X.columns:
        x = X[col].fillna("missing")
        contingency = pd.crosstab(x, y_named)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        rows.append(
            {
                "feature": col,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "cramers_v": cramers_v(x, y_named),
                "n_unique": int(x.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cramers_v", "chi2"], ascending=False).reset_index(drop=True)

def select_top_features(X: pd.DataFrame, y: pd.Series, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = rank_features(X, y)
    selected = ranking.head(top_n)["feature"].tolist()
    return X[selected].copy(), ranking
