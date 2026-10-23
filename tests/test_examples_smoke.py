from __future__ import annotations

import subprocess
import sys

import pytest

from tests.helpers import EXAMPLES_DIR, requires_dl, requires_ml, requires_sklearn_ml


def _run_example(script_name: str, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script_name],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@requires_ml
@pytest.mark.smoke
def test_credit_scoring_xgb_example_runs() -> None:
    result = _run_example("credit_scoring_example.py")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "audit_trail_id" in result.stdout.lower() or "APPROVED" in result.stdout or "DECLINED" in result.stdout


@requires_sklearn_ml
@pytest.mark.smoke
def test_credit_scoring_svm_example_runs() -> None:
    result = _run_example("credit_scoring_svm_example.py")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Downloadable reports" in result.stdout or "audit_report" in result.stdout


@requires_sklearn_ml
@pytest.mark.smoke
def test_wine_multiclass_example_runs() -> None:
    result = _run_example("wine_multiclass_example.py")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Downloadable reports" in result.stdout
    assert "decision_class_2.html" in result.stdout


@requires_ml
@pytest.mark.smoke
def test_shap_model_matrix_example_runs() -> None:
    result = _run_example("shap_model_matrix_example.py", timeout=300)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "--- Summary ---" in result.stdout


@requires_dl
@pytest.mark.smoke
def test_mlp_integrated_gradients_example_runs() -> None:
    result = _run_example("mlp_integrated_gradients_example.py", timeout=240)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Integrated Gradients" in result.stdout or "audit_trail" in result.stdout.lower()


@pytest.mark.smoke
def test_text_highlight_example_runs() -> None:
    pytest.importorskip("torch", reason="text highlight example needs torch (pip install dakma-sdk[dl])")
    pytest.importorskip("captum")
    result = _run_example("text_highlight_example.py", timeout=120)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Downloadable reports" in result.stdout
    assert "text_decision.html" in result.stdout
