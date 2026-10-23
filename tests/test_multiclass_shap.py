"""End-to-end SHAP explanations on multi-class tabular models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import dakma
from tests.helpers import requires_ml, requires_sklearn_ml


def _assert_real_shap_event(event: Any, *, n_features: int) -> None:
    assert event.explanation.status == "success"
    assert event.explanation.method == "shap"
    assert event.explanation.quality.get("used_fallback") is False
    assert "attribution_error" not in event.explanation.metadata
    factors = event.explanation.factors
    assert len(factors) >= 1
    assert all(isinstance(f.impact, float) for f in factors)
    # Fallback uses value * 0.01; real SHAP on trees is rarely exactly that pattern for all factors.
    assert not all(abs(f.impact - float(f.value) * 0.01) < 1e-12 for f in factors)
    # Impacts should be non-trivial on at least one factor for these datasets.
    assert max(abs(f.impact) for f in factors) > 1e-6


@requires_sklearn_ml
def test_random_forest_multiclass_iris_shap_succeeds() -> None:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = RandomForestClassifier(n_estimators=80, random_state=0)
    model.fit(X_train, y_train)

    dm = dakma.init(project="iris-rf-multiclass", regulation="internal-policy", risk_level="high")

    @dm.explain(top_k=3)
    def score(m, row):
        return m.predict_proba(row)

    # Cover a few rows / predicted classes so we don't only hit one layout accidentally.
    for i in (0, 1, 2, 5, 8):
        row = X_test.iloc[[i]]
        proba = model.predict_proba(row)[0]
        event = score(model, row)
        _assert_real_shap_event(event, n_features=X.shape[1])
        # Decision score is the last class probability (SDK convention).
        assert event.decision.score == pytest.approx(float(proba[-1]), abs=1e-6)


@requires_ml
def test_xgboost_multiclass_iris_shap_succeeds() -> None:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = XGBClassifier(
        n_estimators=40,
        max_depth=3,
        eval_metric="mlogloss",
        random_state=0,
    )
    model.fit(X_train, y_train)

    dm = dakma.init(project="iris-xgb-multiclass", regulation="internal-policy", risk_level="high")

    @dm.explain(top_k=3)
    def score(m, row):
        return m.predict_proba(row)

    for i in (0, 1, 2, 5, 8):
        row = X_test.iloc[[i]]
        proba = model.predict_proba(row)[0]
        event = score(model, row)
        _assert_real_shap_event(event, n_features=X.shape[1])
        assert event.decision.score == pytest.approx(float(proba[-1]), abs=1e-6)


@requires_sklearn_ml
def test_local_shap_vector_matches_explainer_multiclass_slice() -> None:
    """Guard the class-axis slice against the shape TreeExplainer returns for RF."""
    import shap
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from dakma_sdk.core import DakmaClient

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X_train, y_train)
    row = X_test.iloc[[0]]
    explanation = shap.Explainer(model)(row)
    raw = np.asarray(explanation.values[0], dtype=float)
    assert raw.ndim == 2 and raw.shape[0] == X.shape[1] and raw.shape[1] == 3

    class_index = DakmaClient._score_class_index(model.predict_proba(row))
    assert class_index == 2  # last class / score index
    sliced = DakmaClient._local_shap_vector(raw, n_features=X.shape[1], class_index=class_index)
    assert sliced.shape == (X.shape[1],)
    assert sliced.tolist() == pytest.approx(raw[:, class_index].tolist())

    factors, err = DakmaClient(project="p", regulation="r", risk_level="high")._top_factors_shap(
        model, row, top_k=3, class_index=class_index
    )
    assert err is None
    assert len(factors) == 3
    # Top factor impact must come from the sliced class column.
    by_abs = sorted(zip(X.columns, raw[:, class_index]), key=lambda t: abs(float(t[1])), reverse=True)
    assert factors[0].name == by_abs[0][0]
    assert factors[0].impact == pytest.approx(float(by_abs[0][1]))
