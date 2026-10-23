"""
Multi-class sample: sklearn ``load_wine`` (3 cultivars), RandomForest, explain one prediction per class.

Test-plan Step 2 ("first multi-class test"). The script records what dakma reports for a 3-class
model and prints an independent SHAP sanity check next to it, so differences are easy to spot.

Run from the `examples/` directory:

    cd examples && python wine_multiclass_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import dakma


dm_w = dakma.init(
    project="wine-cultivar-multiclass",
    regulation="internal-policy",
    risk_level="low",
)


@dm_w.track_data
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # One engineered feature so feature lineage has something to record
    df["phenol_ratio"] = df["flavanoids"] / df["total_phenols"]
    return df


@dm_w.track_training
def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


@dm_w.explain(counterfactual=True)
def classify_wine(model, wine_row):
    return model.predict_proba(wine_row)


def reference_shap_top3(model, row: pd.DataFrame, class_index: int):
    """Top-3 SHAP factors for one class, computed directly (independent of dakma)."""
    values = np.asarray(shap.TreeExplainer(model)(row).values)  # (1, n_features, n_classes)
    per_feature = values[0, :, class_index]
    order = np.argsort(-np.abs(per_feature))[:3]
    return [(row.columns[i], float(per_feature[i])) for i in order]


REPORT_DIR = Path(__file__).resolve().parent / "reports_wine"


if __name__ == "__main__":
    wine = load_wine(as_frame=True)
    class_names = list(wine.target_names)
    raw = wine.frame  # 13 features + "target" (0, 1, 2)

    featured = prepare_features(raw)
    X = featured.drop(columns=["target"])
    y = featured["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = train(X_train, y_train)

    # Global SHAP-based feature importance (shown on decision / audit reports for this model).
    dm_w.register_shap_feature_importance(model, X_train, max_samples=min(300, len(X_train)))

    print("--- Dataset ---")
    print(f"  Train n={len(y_train)}, test n={len(y_test)}, classes={class_names}")
    print(f"  Train class counts: {y_train.value_counts().sort_index().to_dict()}")
    print()

    # --- Hold-out evaluation (multi-class: macro-averaged) ---
    y_pred_test = model.predict(X_test)
    y_t = y_test.to_numpy()
    acc = float(accuracy_score(y_t, y_pred_test))
    prec = float(precision_score(y_t, y_pred_test, average="macro", zero_division=0))
    rec = float(recall_score(y_t, y_pred_test, average="macro", zero_division=0))
    f1v = float(f1_score(y_t, y_pred_test, average="macro", zero_division=0))
    auc = float(roc_auc_score(y_t, model.predict_proba(X_test), multi_class="ovr", average="macro"))
    cm_test = confusion_matrix(y_t, y_pred_test)

    # monitor() assumes 0/1 labels; with 3 classes "positive_prediction_rate" is the mean class id.
    monitor_report = dm_w.monitor(y_true=y_t, y_pred=y_pred_test)

    dm_w.register_governance(
        intended_use="Demo multi-class classification of wine cultivar from chemical analysis.",
        known_limitations="Tiny public dataset (178 rows); for SDK testing only.",
        human_oversight="Not used for decisions; explainability test harness.",
        data_provenance="sklearn.datasets.load_wine (UCI Wine recognition dataset).",
        model_changelog=[
            {"version": "demo-1.0", "date": "2026-10-19", "notes": "RandomForest + SHAP multi-class test"},
        ],
    )
    dm_w.register_evaluation(
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        split_description="70/30 stratified random split (sklearn train_test_split, random_state=42)",
        test_metrics={
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1v,
            "roc_auc_ovr_macro": auc,
            "accuracy": acc,
        },
        confusion_matrix=cm_test.tolist(),
    )

    print("--- monitor() ---")
    for key, value in monitor_report.items():
        print(f"  {key}: {value}")
    print(f"  (sklearn) macro F1 = {f1v:.3f}, confusion matrix = {cm_test.tolist()}")
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decision_reports = []

    # Explain one test row for each true class, so Decision.value can be compared across classes 0, 1, 2.
    for cls in range(len(class_names)):
        wine_row = X_test[y_test == cls].iloc[[0]]
        proba = model.predict_proba(wine_row)[0]
        predicted = int(np.argmax(proba))

        result = classify_wine(model, wine_row)

        print(f"=== True class {cls} ({class_names[cls]}) — row index {wine_row.index[0]} ===")
        print(f"  predict_proba            : {np.round(proba, 3).tolist()}  -> predicted class {predicted}")
        print(f"  Decision.value           : {result.decision.value}")
        print(f"  Decision.score           : {result.decision.score:.3f}  (threshold {result.decision.threshold})")
        print(f"  Explanation.status       : {result.explanation.status}")
        print(f"  Explanation.counterfactual: {result.explanation.counterfactual}")
        print("  dakma top factors        :")
        for f in result.explanation.factors:
            print(f"      {f.name:<22} value={f.value:<10.3f} impact={f.impact:+.4f} ({f.direction})")
        print(f"  SHAP top factors (class {predicted}, computed directly):")
        for name, impact in reference_shap_top3(model, wine_row, predicted):
            print(f"      {name:<22} value={float(wine_row.iloc[0][name]):<10.3f} impact={impact:+.4f}")
        print(f"  audit_trail_id: {result.explanation.audit_trail_id}")
        print()

        decision_reports.append(
            result.write_report(
                REPORT_DIR / f"decision_class_{cls}.html",
                title=f"Wine decision — true class {cls} ({class_names[cls]})",
                format="html",
            )
        )

    print("--- Sanity check: global importance ---")
    fi = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
    print("  model.feature_importances_ (top 5):")
    for name, v in fi:
        print(f"      {name:<22} {v:.4f}")
    print("  dakma register_shap_feature_importance (top 5):")
    for row in dm_w._shap_fi_rows[:5]:
        print(f"      {row.name:<22} {row.mean_abs_shap:.4f}")
    print()

    print("--- Full audit trail (all steps, tabular) ---")
    print(dm_w.format_audit_log_markdown())

    audit_md = dm_w.write_audit_report(REPORT_DIR / "audit_report.md", title="Wine multi-class — audit trail")
    audit_html = dm_w.write_audit_report(
        REPORT_DIR / "audit_report.html", title="Wine multi-class — audit trail", format="html"
    )
    print()
    print("--- Downloadable reports written ---")
    print(f"  {audit_md}")
    print(f"  {audit_html}")
    for path in decision_reports:
        print(f"  {path}")
