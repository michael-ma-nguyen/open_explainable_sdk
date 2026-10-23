"""
Which model / class-count combinations does dakma's per-prediction SHAP explanation handle?

Trains several classifiers on a binary dataset (``load_breast_cancer``) and a 3-class dataset
(``load_wine``), explains the first test row with ``@dk.explain()``, and compares dakma's top
factors with SHAP computed directly. dakma silently falls back to a ``value * 0.01`` heuristic
(or ``coef * value`` for linear models) when its SHAP step fails, so each row also reports
whether the top factor matches that fallback.

Run from the `examples/` directory:

    cd examples && python shap_model_matrix_example.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import dakma

warnings.filterwarnings("ignore")

dk = dakma.init(project="shap-model-matrix")


@dk.explain()
def predict(model, row):
    return model.predict_proba(row)


def make_models():
    return {
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=5000),
    }


def reference_shap(model, X_background: pd.DataFrame, row: pd.DataFrame, class_index: int):
    """SHAP values for one row and one class, computed directly (independent of dakma)."""
    explanation = shap.Explainer(model.predict_proba, X_background, seed=0)(row)
    values = np.asarray(explanation.values)  # (1, n_features, n_classes) for predict_proba
    return values[0, :, class_index]


def fallback_impacts(model, row: pd.DataFrame) -> np.ndarray:
    """What dakma's ``_top_factors_fallback`` would produce for this row."""
    values = row.iloc[0].to_numpy(dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
        # Multi-class linear models have one coef row per class, so dakma leaves every impact at 0.
        return coef * values if len(coef) == len(values) else np.zeros_like(values)
    return values * 0.01


def check(dataset_name: str, loader) -> list:
    X, y = loader(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    background = X_train.sample(n=min(100, len(X_train)), random_state=0)
    row = X_test.iloc[[0]]
    rows = []

    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        # Same call dakma makes internally (no background data passed).
        try:
            raw_shape = str(np.asarray(shap.Explainer(model)(row).values).shape)
        except Exception as exc:
            raw_shape = f"error: {type(exc).__name__}"
        proba = model.predict_proba(row)[0]
        predicted = int(np.argmax(proba))
        # Binary: dakma scores and explains class 1, so compare against class 1.
        # Multi-class: dakma has no class choice for SHAP, so compare against the predicted class.
        ref_class = 1 if len(proba) == 2 else predicted

        event = predict(model, row)
        dakma_top = event.explanation.factors[0]
        ref = reference_shap(model, background, row, ref_class)
        ref_order = np.argsort(-np.abs(ref))
        ref_top = row.columns[ref_order[0]]

        fb = fallback_impacts(model, row)
        used_fallback = bool(np.isclose(dakma_top.impact, fb[list(row.columns).index(dakma_top.name)]))

        print(f"=== {dataset_name} | {model_name} ===")
        print(f"  SHAP output (dakma's call): {raw_shape}")
        print(f"  predict_proba             : {np.round(proba, 3).tolist()} -> class {predicted}")
        print(f"  Decision                  : {event.decision.value} (score {event.decision.score:.3f})")
        print(f"  Explanation.status        : {event.explanation.status}")
        print("  dakma top 3               : " + ", ".join(f"{f.name} {f.impact:+.3f}" for f in event.explanation.factors))
        print(
            f"  SHAP top 3 (class {ref_class})    : "
            + ", ".join(f"{row.columns[i]} {ref[i]:+.3f}" for i in ref_order[:3])
        )
        verdict = "FALLBACK (not SHAP)" if used_fallback else "real SHAP"
        print(f"  dakma used                : {verdict}")
        print()
        rows.append((dataset_name, model_name, raw_shape, dakma_top.name, ref_top, verdict))
    return rows


if __name__ == "__main__":
    results = check("binary (breast_cancer)", load_breast_cancer)
    results += check("3-class (wine)", load_wine)

    print("--- Summary ---")
    header = ("Dataset", "Model", "SHAP shape", "dakma top", "SHAP top", "dakma used")
    widths = [24, 20, 20, 26, 26, 20]
    print("  " + "".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in results:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(r, widths)))
