"""Human-readable tabular formatting for audit log entries (Markdown tables + HTML reports)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

DOCUMENTATION_DISCLAIMER_MARKDOWN = (
    "> **Disclaimer:** dakma-sdk helps you *document* model lifecycle and explainability "
    "information. It does **not** provide legal advice or regulatory certification."
)

DOCUMENTATION_DISCLAIMER_HTML = (
    '<p class="dakma-disclaimer"><strong>Disclaimer:</strong> dakma-sdk helps you '
    "<em>document</em> model lifecycle and explainability information. It does "
    "<strong>not</strong> provide legal advice or regulatory certification.</p>"
)


def _decision_rows(dec_out: Dict[str, Any]) -> List[Any]:
    """Field/value rows for the Decision table (multi-class decisions have no threshold)."""
    try:
        sc = float(dec_out.get("score")) if dec_out.get("score") is not None else None
        th = float(dec_out.get("threshold")) if dec_out.get("threshold") is not None else None
        score_s = f"{sc:.2f}" if sc is not None else "—"
        th_s = f"{th:.2f}" if th is not None else "—"
    except (TypeError, ValueError):
        score_s = _cell(dec_out.get("score"))
        th_s = _cell(dec_out.get("threshold"))
    if dec_out.get("predicted_class") is not None:
        return [
            ("predicted class", dec_out.get("predicted_class")),
            ("class index", dec_out.get("class_index")),
            ("probability", score_s),
        ]
    return [("label", dec_out.get("label")), ("score", score_s), ("threshold", th_s)]


def _cell(value: Any, *, max_len: int = 500) -> str:
    if value is None:
        return "—"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        text = repr(value)
    else:
        text = str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text.replace("|", "¦")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [h, sep]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _kv_table(pairs: List[tuple[str, Any]], *, title: Optional[str] = None) -> str:
    parts: List[str] = []
    if title:
        parts.append(f"**{title}**")
    parts.append(_markdown_table(["Field", "Value"], pairs))
    return "\n".join(parts)


def _schema_to_rows(schema: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for col in sorted(schema.keys()):
        info = schema[col]
        if isinstance(info, dict):
            rows.append([col, info.get("dtype", "—"), info.get("nulls", "—")])
        else:
            rows.append([col, _cell(info), "—"])
    return rows


# --- Governance + evaluation (all audit and decision reports) ---

_GOVERNANCE_FIELD_ROWS: List[Tuple[str, str]] = [
    ("intended_use", "Intended use"),
    ("known_limitations", "Known limitations"),
    ("human_oversight", "Human oversight mechanism"),
    ("data_provenance", "Data source / provenance (training data)"),
    ("model_changelog", "Model version history / change log"),
]

def _format_changelog_value(value: Any) -> str:
    if value is None:
        return "— *not documented*"
    if isinstance(value, str):
        return _cell(value, max_len=2000)
    if isinstance(value, list):
        parts: List[str] = []
        for i, item in enumerate(value):
            if isinstance(item, dict):
                ver = item.get("version", item.get("v", f"entry {i + 1}"))
                when = item.get("date", item.get("when", ""))
                notes = item.get("notes", item.get("text", item))
                parts.append(
                    f"- **{str(ver).replace('**', '')}** ({_cell(when, max_len=200)}): "
                    f"{_cell(notes, max_len=800)}"
                )
            else:
                parts.append(f"- {_cell(item, max_len=2000)}")
        return "\n".join(parts) if parts else "— *not documented*"
    return _cell(value, max_len=2000)


def format_governance_markdown(governance: Optional[Mapping[str, Any]] = None) -> str:
    g = dict(governance or {})
    std_keys = {k for k, _ in _GOVERNANCE_FIELD_ROWS}
    lines: List[str] = [
        "## Governance documentation",
        "",
        "Fields below are a *documentation template* only. Use "
        "``DakmaClient.register_governance(...)`` to fill them. Empty items are flagged in reports.",
        "",
    ]
    rows: List[Sequence[Any]] = []
    for key, label in _GOVERNANCE_FIELD_ROWS:
        v = g.get(key)
        if key == "model_changelog" and (v is not None and v != ""):
            val = _format_changelog_value(v)
        else:
            val = "— *not documented*" if v is None or v == "" else _cell(v, max_len=2000)
        rows.append((label, val))
    for k in sorted(x for x in g if x not in std_keys):
        val = g.get(k)
        rows.append((k, "— *not documented*" if val is None or val == "" else _cell(val, max_len=2000)))
    lines.append(_markdown_table(["Topic", "Documentation"], rows))
    return "\n".join(lines)


def _training_entries_only_report_training_score(
    entries: Optional[Sequence[Dict[str, Any]]],
) -> bool:
    if not entries:
        return False
    for e in entries:
        if e.get("type") != "training_tracking":
            continue
        m = e.get("metrics") or {}
        if not m:
            continue
        keys = {k for k, v in m.items() if v is not None}
        if keys and keys == {"training_score"}:
            return True
    return False


def _test_metric_keys_match(m: Dict[str, Any], *needles: str) -> bool:
    for n in needles:
        for k in m:
            if str(k).lower() == n or n in str(k).lower().replace(" ", ""):
                return True
    return False


def _evaluation_gap_bullets(
    ev: Dict[str, Any],
    test_metrics: Dict[str, Any],
    *,
    entries: Optional[Sequence[Dict[str, Any]]],
) -> List[str]:
    n_train, n_test = ev.get("n_train"), ev.get("n_test")
    cm = ev.get("confusion_matrix")
    gaps: List[str] = []
    if not ev.get("split_description"):
        gaps.append("No train/validation/test methodology described (`split_description`).")
    if n_train is None and n_test is None:
        gaps.append("No dataset size (`n_train` / `n_test`) provided.")
    if not test_metrics:
        gaps.append(
            "No **test-set** metrics — training-only or in-sample scores are not a substitute for hold-out evaluation."
        )
    if cm is None or cm == "":
        gaps.append("No confusion matrix on hold-out data (`confusion_matrix`).")
    if test_metrics and not _test_metric_keys_match(test_metrics, "roc_auc", "roc_auc_score", "auc"):
        if not any("roc" in str(k).lower() or "auc" in str(k).lower() for k in test_metrics):
            gaps.append("ROC-AUC (or a documented separation metric) is missing from `test_metrics`.")
    if _training_entries_only_report_training_score(entries) and not test_metrics:
        gaps.append(
            "Training log includes only `training_score` (e.g. in-sample) — a perfect score is a **red flag** for "
            "overfitting; add a held-out test set, `test_metrics`, and `split_description`."
        )
    seen: set = set()
    out: List[str] = []
    for g in gaps:
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def format_evaluation_block_markdown(
    evaluation: Optional[Mapping[str, Any]] = None,
    *,
    entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    ev = dict(evaluation or {})
    test_metrics: Dict[str, Any] = dict(ev.get("test_metrics") or {})
    n_train, n_test = ev.get("n_train"), ev.get("n_test")
    split_d = ev.get("split_description")

    lines: List[str] = [
        "## Evaluation and dataset",
        "",
        "### Dataset and train/test split",
        "",
    ]
    lines.append(
        _markdown_table(
            ["Item", "Value"],
            [
                (
                    "Split / methodology description",
                    "— *not provided*" if not split_d else _cell(str(split_d), max_len=2000),
                ),
                (
                    "Training set size (n_train)",
                    n_train if n_train is not None else "— *not provided*",
                ),
                (
                    "Test / hold-out set size (n_test)",
                    n_test if n_test is not None else "— *not provided*",
                ),
            ],
        )
    )
    lines.append("")
    lines.append("### Test-set / hold-out metrics")
    lines.append(
        "Report metrics on **held-out** data (not training fit / training_score only). Use "
        "``DakmaClient.register_evaluation(test_metrics={...}, ...)``."
    )
    lines.append("")

    if test_metrics:
        rows = [[k, _cell(v, max_len=2000)] for k, v in sorted(test_metrics.items())]
    else:
        rows = [("(none specified)", "— *not provided*")]
    lines.append(_markdown_table(["Metric (test/hold-out)", "Value"], rows))

    lines.append("")
    lines.append("### Checklist: recommended test metrics")
    want_display = [
        ("Precision (test)", "precision"),
        ("Recall (test)", "recall"),
        ("F1 (test)", "f1", "f1_score"),
        ("ROC-AUC (test)", "roc_auc", "roc_auc_score", "auc"),
        ("Accuracy (test)", "accuracy", "test_accuracy"),
    ]
    check_rows: List[Sequence[Any]] = []
    for spec in want_display:
        label, *keys = spec
        ok = _test_metric_keys_match(test_metrics, *keys) if test_metrics else False
        st = "✓" if ok else "— *not in test_metrics*"
        check_rows.append((label, st))
    lines.append(_markdown_table(["Item", "Status"], check_rows))

    lines.append("")
    lines.append("### Confusion matrix (test / hold-out)")
    cm = ev.get("confusion_matrix")
    if cm is not None and cm != "":
        try:
            arr = np.asarray(cm)
            if arr.shape == (2, 2):
                lines.append(
                    _markdown_table(
                        ["", "Pred 0", "Pred 1"],
                        [
                            ("Actual 0", int(arr[0, 0]), int(arr[0, 1])),
                            ("Actual 1", int(arr[1, 0]), int(arr[1, 1])),
                        ],
                    )
                )
            else:
                lines.append(_cell(cm, max_len=2000))
        except (TypeError, ValueError, IndexError):
            lines.append(_cell(cm, max_len=2000))
    else:
        lines.append("— *not provided* (pass ``confusion_matrix=`` to ``register_evaluation``)")

    lines.append("")
    lines.append("### Gaps to address (automated notes)")
    gap_bullets = _evaluation_gap_bullets(ev, test_metrics, entries=entries)
    if not gap_bullets:
        lines.append(
            "- *No automated gap flags (still verify precision/recall/F1, lineage, and oversight manually).*"
        )
    else:
        for g in gap_bullets:
            lines.append(f"- {g}")

    return "\n".join(lines)


def format_compliance_preamble_markdown(
    *,
    governance: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
    entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    g = format_governance_markdown(governance)
    e = format_evaluation_block_markdown(evaluation, entries=entries)
    return f"{DOCUMENTATION_DISCLAIMER_MARKDOWN}\n\n{g}\n\n{e}\n\n---\n\n"


def format_audit_entry_markdown(entry: Dict[str, Any], index: int) -> str:
    """Render a single audit log entry as titled Markdown sections with tables."""
    etype = entry.get("type", "unknown")
    title = f"### {index}. {etype}"
    lines: List[str] = [title]

    if etype == "data_tracking":
        lines.append("")
        lines.append(
            _kv_table(
                [
                    ("timestamp", entry.get("timestamp")),
                    ("function", entry.get("function")),
                ]
            )
        )
        inp = entry.get("input_schema") or {}
        out = entry.get("output_schema") or {}
        lin = entry.get("lineage_added") or {}
        if inp:
            lines.append("")
            lines.append("**Input schema**")
            lines.append(_markdown_table(["Column", "dtype", "nulls"], _schema_to_rows(inp)))
        if out:
            lines.append("")
            lines.append("**Output schema**")
            lines.append(_markdown_table(["Column", "dtype", "nulls"], _schema_to_rows(out)))
        if lin:
            lines.append("")
            lines.append("**Feature lineage (new/updated columns)**")
            rows = [[k, _cell(v)] for k, v in sorted(lin.items())]
            lines.append(_markdown_table(["Feature", "Expression / source"], rows))
        return "\n".join(lines)

    if etype == "training_tracking":
        lines.append("")
        lines.append(
            _kv_table(
                [
                    ("timestamp", entry.get("timestamp")),
                    ("function", entry.get("function")),
                    ("model_class", entry.get("model_class")),
                    ("started_at", entry.get("started_at")),
                    ("finished_at", entry.get("finished_at")),
                ]
            )
        )
        params = entry.get("model_params") or {}
        if params:
            lines.append("")
            lines.append("**Model parameters**")
            pr = [[k, _cell(v)] for k, v in sorted(params.items())]
            lines.append(_markdown_table(["Parameter", "Value"], pr))
        metrics = entry.get("metrics") or {}
        if metrics:
            lines.append("")
            lines.append("**Metrics**")
            mr = [[k, _cell(v)] for k, v in sorted(metrics.items())]
            lines.append(_markdown_table(["Metric", "Value"], mr))
        cu = entry.get("compute_usage")
        if cu and isinstance(cu, dict):
            lines.append("")
            lines.append(_format_compute_usage_markdown(cu))
        return "\n".join(lines)

    if etype == "monitoring":
        lines.append("")
        payload = entry.get("payload") or {}
        lines.append(_kv_table([("timestamp", entry.get("timestamp"))]))
        lines.append("")
        lines.append("**Monitoring summary**")
        acc = payload.get("accuracy")
        ppr = payload.get("positive_prediction_rate")
        drift = payload.get("drift_from_expected_rate")
        summary_rows = [
            ("accuracy", acc),
            ("positive_prediction_rate", ppr),
            ("drift_from_expected_rate", drift if drift is not None else "—"),
        ]
        lines.append(_markdown_table(["Metric", "Value"], summary_rows))
        groups = payload.get("group_positive_rates") or {}
        if groups:
            lines.append("")
            lines.append("**Group positive prediction rates** (bias / fairness signal)")
            gr = [[g, _cell(r)] for g, r in sorted(groups.items())]
            lines.append(_markdown_table(["Group", "Mean predicted positive"], gr))
        return "\n".join(lines)

    if etype == "inference_explain":
        lines.append("")
        lines.append(_kv_table([("timestamp", entry.get("timestamp")), ("function", entry.get("function"))]))
        res = entry.get("result") or {}
        lines.append("")
        lines.append(format_inference_result_markdown(res, include_compliance_sections=False))
        return "\n".join(lines)

    # Fallback: generic key-value
    lines.append("")
    skip = {"type"}
    pairs = [(k, entry[k]) for k in sorted(entry.keys()) if k not in skip]
    lines.append(_markdown_table(["Field", "Value"], pairs))
    return "\n".join(lines)


def _format_compute_usage_markdown(cu: Mapping[str, Any]) -> str:
    """Markdown table for a :class:`~dakma_sdk.compute_usage.ComputeUsageSnapshot` dict."""
    lines: List[str] = ["**Compute usage** (this process)"]
    w = cu.get("wall_time_ms")
    cpu = cu.get("process_cpu_time_ms")
    rs = cu.get("process_ram_start_mb")
    re = cu.get("process_ram_end_mb")
    rd = cu.get("process_ram_delta_mb")
    gp = cu.get("gpu_peak_alloc_mb")
    gd = cu.get("gpu_device")
    rows = [
        ("Wall time", f"{w} ms" if w is not None else "—"),
        ("CPU time (user + system)", f"{cpu:.2f} ms" if isinstance(cpu, (int, float)) else "—"),
        ("RAM RSS (start)", f"{rs:.2f} MB" if isinstance(rs, (int, float)) else "—"),
        ("RAM RSS (end)", f"{re:.2f} MB" if isinstance(re, (int, float)) else "—"),
        ("RAM delta", f"{rd:+.3f} MB" if isinstance(rd, (int, float)) else "—"),
        ("GPU peak memory allocated", f"{gp:.2f} MB" if isinstance(gp, (int, float)) else "—"),
        ("GPU device", _cell(gd) if gd else "—"),
    ]
    lines.append(_markdown_table(["Metric", "Value"], rows))
    lines.append("")
    lines.append(
        "*CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available.*"
    )
    return "\n".join(lines)


def _format_compute_usage_html(cu: Mapping[str, Any]) -> str:
    parts: List[str] = [_html_h3("Compute usage (this process)")]
    w = cu.get("wall_time_ms")
    cpu = cu.get("process_cpu_time_ms")
    rs = cu.get("process_ram_start_mb")
    re = cu.get("process_ram_end_mb")
    rd = cu.get("process_ram_delta_mb")
    gp = cu.get("gpu_peak_alloc_mb")
    gd = cu.get("gpu_device")
    rows = [
        ("Wall time", f"{w} ms" if w is not None else "—"),
        ("CPU time (user + system)", f"{cpu:.2f} ms" if isinstance(cpu, (int, float)) else "—"),
        ("RAM RSS (start)", f"{rs:.2f} MB" if isinstance(rs, (int, float)) else "—"),
        ("RAM RSS (end)", f"{re:.2f} MB" if isinstance(re, (int, float)) else "—"),
        ("RAM delta", f"{rd:+.3f} MB" if isinstance(rd, (int, float)) else "—"),
        ("GPU peak memory allocated", f"{gp:.2f} MB" if isinstance(gp, (int, float)) else "—"),
        ("GPU device", _cell(gd) if gd else "—"),
    ]
    parts.append(_html_table(["Metric", "Value"], rows))
    parts.append(_html_p_em("CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available."))
    return "\n".join(parts)


def format_inference_result_markdown(
    result_dict: Dict[str, Any],
    *,
    include_compliance_sections: bool = True,
) -> str:
    """Format a serialized EnrichedResult (from ``asdict``) as readable tables."""
    parts: List[str] = []
    explain_pre = result_dict.get("explain") or {}
    if include_compliance_sections:
        g = explain_pre.get("governance") or {}
        e = explain_pre.get("evaluation") or {}
        parts.append(format_compliance_preamble_markdown(governance=g, evaluation=e, entries=None))
    dec_out = result_dict.get("decision_output") or {}
    explain = result_dict.get("explain") or {}
    raw_decision = result_dict.get("decision")

    parts.append("**Decision**")
    decision_rows = _decision_rows(dec_out)
    parts.append(
        _markdown_table(
            ["Field", "Value"],
            decision_rows,
        )
    )
    parts.append("")
    parts.append("**Raw model output** (unchanged)")
    parts.append(_markdown_table(["decision"], [[_cell(raw_decision)]]))

    parts.append("")
    parts.append("**Explainability**")
    meta = explain.get("metadata") or {}
    quality = explain.get("quality") or {}
    exp_rows = [
        ("audit_trail_id", explain.get("audit_trail_id")),
        ("model_version", explain.get("model_version")),
        (
            "attribution",
            explain.get("attribution_backend") or explain.get("method") or "shap",
        ),
        ("status", explain.get("status") or "—"),
        ("plain_language", explain.get("plain_language")),
        ("counterfactual", explain.get("counterfactual") or "—"),
    ]
    if meta.get("attribution_error"):
        exp_rows.append(("attribution_error", meta.get("attribution_error")))
    if quality.get("used_fallback"):
        exp_rows.append(("used_fallback", True))
    parts.append(_markdown_table(["Field", "Value"], exp_rows))

    flags = explain.get("regulation_flags") or []
    if flags:
        parts.append("")
        parts.append("**Documentation status**")
        parts.append(_markdown_table(["Note"], [[f] for f in flags]))

    cu = explain.get("compute_usage")
    if cu and isinstance(cu, dict):
        parts.append("")
        parts.append(_format_compute_usage_markdown(cu))

    lh, ga, gf, la = _attribution_labels(explain)
    tokens = explain.get("text_attributions") or []
    if tokens:
        highlight = format_text_highlight_markdown(tokens)
        if highlight:
            parts.append("")
            parts.append("**Highlighted text (this decision)**")
            parts.append("")
            parts.append(highlight)

    factors = explain.get("top_factors") or []
    if factors:
        parts.append("")
        parts.append(f"**{lh}**")
        chart = _html_shap_local_chart(factors, axis_label=la)
        if chart:
            parts.append("")
            parts.append(chart)

    fi_list = explain.get("feature_importance") or []
    if fi_list:
        parts.append("")
        parts.append("**Feature importance (global)**")
        parts.append(gf)
        chart = _html_shap_global_chart(fi_list, axis_label=ga)
        if chart:
            parts.append("")
            parts.append(chart)

    meta = explain.get("metadata") or {}
    if meta:
        parts.append("")
        parts.append("**Metadata**")
        mr = [[k, _cell(v)] for k, v in sorted(meta.items()) if k != "feature_lineage"]
        parts.append(_markdown_table(["Key", "Value"], mr))
        fl = meta.get("feature_lineage") or {}
        if fl:
            parts.append("")
            parts.append("**Feature lineage (snapshot)**")
            lr = [[k, _cell(v)] for k, v in sorted(fl.items())]
            parts.append(_markdown_table(["Feature", "Expression"], lr))

    return "\n".join(parts)


def format_audit_log_markdown(
    entries: Sequence[Dict[str, Any]],
    *,
    title: str = "Audit trail",
    governance: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format a full audit log as one Markdown document with numbered sections."""
    preamble = format_compliance_preamble_markdown(
        governance=governance,
        evaluation=evaluation,
        entries=entries,
    )
    lines: List[str] = [f"## {title}", "", preamble, f"*Total entries: {len(entries)}*", ""]
    for i, entry in enumerate(entries, start=1):
        lines.append(format_audit_entry_markdown(entry, i))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- HTML (downloadable / browser-friendly reports) ---


def _html_esc(value: Any) -> str:
    return html.escape(_cell(value), quote=False)


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    th = "".join(f"<th>{html.escape(str(h), quote=False)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{_html_esc(c)}</td>" for c in row)
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        '<table class="dakma-table">\n'
        f"<thead><tr>{th}</tr></thead>\n"
        f"<tbody>\n{chr(10).join(body_rows)}\n</tbody>\n"
        "</table>"
    )


def _html_table_two_col_doc_second_may_be_markup(rows: Sequence[Tuple[str, Any]]) -> str:
    """Two-column table; second cell may contain pre-rendered HTML (e.g. `<ul>...</ul>`)."""
    body_rows = []
    for label, val in rows:
        raw = isinstance(val, str) and val.lstrip().startswith("<")
        cell = val if raw else _html_esc(val)
        body_rows.append(
            f"<tr><th>{html.escape(str(label), quote=False)}</th><td>{cell}</td></tr>"
        )
    return (
        '<table class="dakma-table">\n<tbody>\n'
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>"
    )


def _html_h2(text: str) -> str:
    return f"<h2>{html.escape(text, quote=False)}</h2>"


def _html_h3(text: str) -> str:
    return f"<h3>{html.escape(text, quote=False)}</h3>"


def _html_p_em(text: str) -> str:
    return f"<p><em>{html.escape(text, quote=False)}</em></p>"


_MAX_SHAP_CHART_ROWS = 20


def _attribution_labels(explain: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    """Headings and chart axis copy: local heading, global bar axis label, global footnote (markdown), local axis."""
    if explain.get("attribution_backend") == "integrated_gradients":
        return (
            "Top factors (Integrated Gradients for this row)",
            "Mean |IG|",
            "*Mean |Integrated Gradients| averaged over the reference sample passed to "
            "register_integrated_gradients_feature_importance.*",
            "Integrated Gradients (this row)",
        )
    return (
        "Top factors (SHAP for this row)",
        "Mean |SHAP|",
        "*Mean |SHAP| averaged over the reference sample passed to register_shap_feature_importance.*",
        "SHAP (this row)",
    )


def _truncate_chart_label(s: str, max_len: int = 34) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _svg_text_escape(s: str) -> str:
    return html.escape(s.replace("\n", " ").replace("\r", ""), quote=False)


def _pairs_global_importance(fi_list: Sequence[Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for row in fi_list:
        if isinstance(row, dict):
            rows.append((str(row.get("name", "")), float(row.get("mean_abs_shap") or 0.0)))
    return rows


def _pairs_local_impacts(factors: Sequence[Any]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for tf in factors:
        if isinstance(tf, dict):
            rows.append((str(tf.get("name", "")), float(tf.get("impact") or 0.0)))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def _svg_global_importance_bars(
    rows: Sequence[Tuple[str, float]],
    *,
    width: int = 560,
    axis_label: str = "Mean |SHAP|",
) -> str:
    rows = list(rows[:_MAX_SHAP_CHART_ROWS])
    if not rows:
        return ""
    margin_l, margin_r, row_h, pad_top = 132, 72, 26, 14
    bar_max = float(width - margin_l - margin_r)
    n = len(rows)
    height = pad_top + n * row_h + 10
    vmax = max(v for _, v in rows) or 1e-12
    axis_esc = html.escape(axis_label, quote=False)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<text x="{margin_l}" y="{pad_top - 4}" font-size="11" fill="#666">{axis_esc}</text>',
    ]
    for i, (name, val) in enumerate(rows):
        y = pad_top + i * row_h
        lab = _truncate_chart_label(name)
        bw = (val / vmax) * bar_max
        parts.append(f'<text x="0" y="{y + 16}" font-size="12" fill="#1a1a1a">{_svg_text_escape(lab)}</text>')
        parts.append(
            f'<rect x="{margin_l}" y="{y + 2}" width="{max(bw, 0.5):.1f}" height="18" '
            f'fill="#2563eb" rx="2" opacity="0.92">'
            f"<title>{_svg_text_escape(str(name))}: {val:.6f}</title></rect>"
        )
        parts.append(f'<text x="{margin_l + bw + 4}" y="{y + 16}" font-size="11" fill="#444">{val:.4f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_signed_shap_bars(
    rows: Sequence[Tuple[str, float]],
    *,
    width: int = 560,
    axis_label: str = "SHAP (this row)",
) -> str:
    rows = list(rows[:_MAX_SHAP_CHART_ROWS])
    if not rows:
        return ""
    margin_l, margin_r, row_h, pad_top = 132, 88, 26, 18
    inner = float(width - margin_l - margin_r)
    half = inner / 2.0
    zero_x = margin_l + half
    vmax = max(abs(v) for _, v in rows) or 1e-12
    scale = half / vmax
    n = len(rows)
    height = pad_top + n * row_h + 10
    axis_esc = html.escape(axis_label, quote=False)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<line x1="{zero_x:.1f}" y1="{pad_top - 8}" x2="{zero_x:.1f}" y2="{height - 4}" stroke="#ccc" stroke-width="1"/>',
        f'<text x="{margin_l}" y="{pad_top - 6}" font-size="11" fill="#666">{axis_esc}</text>',
    ]
    for i, (name, v) in enumerate(rows):
        y = pad_top + i * row_h
        lab = _truncate_chart_label(name)
        parts.append(f'<text x="0" y="{y + 16}" font-size="12" fill="#1a1a1a">{_svg_text_escape(lab)}</text>')
        px = float(v) * scale
        if v >= 0:
            x0, w_rect = zero_x, px
        else:
            x0, w_rect = zero_x + px, -px
        color = "#1d4ed8" if v >= 0 else "#b91c1c"
        parts.append(
            f'<rect x="{x0:.1f}" y="{y + 2}" width="{max(w_rect, 0.5):.1f}" height="18" '
            f'fill="{color}" rx="2" opacity="0.92">'
            f"<title>{_svg_text_escape(str(name))}: {v:+.6f}</title></rect>"
        )
        parts.append(
            f'<text x="{margin_l + inner + 4}" y="{y + 16}" font-size="11" fill="#444">{v:+.4f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _html_shap_global_chart(
    fi_list: Sequence[Any],
    *,
    axis_label: str = "Mean |SHAP|",
) -> str:
    rows_all = _pairs_global_importance(fi_list)
    svg = _svg_global_importance_bars(rows_all, axis_label=axis_label)
    if not svg:
        return ""
    note = ""
    if len(rows_all) > _MAX_SHAP_CHART_ROWS:
        note = f'<p class="dakma-chart-note"><em>Showing top {_MAX_SHAP_CHART_ROWS} features.</em></p>'
    return f'<figure class="dakma-shap-chart">{svg}</figure>{note}'


def _html_shap_local_chart(
    factors: Sequence[Any],
    *,
    axis_label: str = "SHAP (this row)",
) -> str:
    rows = _pairs_local_impacts(factors)
    svg = _svg_signed_shap_bars(rows, axis_label=axis_label)
    if not svg:
        return ""
    return f'<figure class="dakma-shap-chart">{svg}</figure>'


_MAX_HIGHLIGHT_TOKENS = 300
_HIGHLIGHT_UP_RGB = (29, 78, 216)  # #1d4ed8, same blue as the signed impact bars
_HIGHLIGHT_DOWN_RGB = (185, 28, 28)  # #b91c1c, same red as the signed impact bars
_HIGHLIGHT_LEGEND = (
    "Shading shows each token's Integrated Gradients impact: blue pushed the score up, "
    "red pushed it down, stronger colour means larger impact."
)


def _pairs_token_impacts(attrs: Sequence[Any]) -> List[Tuple[str, float]]:
    """Normalize token attributions (dataclasses or report dicts) to ``(token, impact)``."""
    rows: List[Tuple[str, float]] = []
    for item in attrs:
        if isinstance(item, Mapping):
            token, impact = item.get("token", ""), item.get("impact")
        else:
            token, impact = getattr(item, "token", ""), getattr(item, "impact", None)
        try:
            rows.append((str(token), float(impact or 0.0)))
        except (TypeError, ValueError):
            rows.append((str(token), 0.0))
    return rows


def _token_shade(impact: float, vmax: float) -> Tuple[Tuple[int, int, int], float]:
    """Colour and opacity for one token, scaled against the strongest impact in the text."""
    rgb = _HIGHLIGHT_UP_RGB if impact >= 0 else _HIGHLIGHT_DOWN_RGB
    alpha = min(abs(impact) / vmax, 1.0) * 0.85 if vmax else 0.0
    return rgb, alpha


def _svg_text_highlight(attrs: Sequence[Any], *, width: int = 560) -> str:
    rows = _pairs_token_impacts(attrs)[:_MAX_HIGHLIGHT_TOKENS]
    if not rows:
        return ""
    char_w, font_size, pad_x, gap, line_h, pad_top = 7.0, 12, 5.0, 4.0, 24.0, 16.0
    vmax = max(abs(v) for _, v in rows) or 1e-12

    placed: List[Tuple[float, float, float, str, float]] = []  # x, y, box width, token, impact
    x, line = 0.0, 0
    for token, impact in rows:
        box_w = len(token) * char_w + 2 * pad_x
        if x > 0 and x + box_w > width:
            x, line = 0.0, line + 1
        placed.append((x, pad_top + line * line_h, box_w, token, impact))
        x += box_w + gap
    height = pad_top + (line + 1) * line_h + 6

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
        f'viewBox="0 0 {width} {height:.0f}" role="img">',
        f'<text x="0" y="11" font-size="11" fill="#666">Integrated Gradients (per token)</text>',
    ]
    for bx, by, box_w, token, impact in placed:
        (r, g, b), alpha = _token_shade(impact, vmax)
        text_fill = "#ffffff" if alpha > 0.5 else "#1a1a1a"
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{box_w:.1f}" height="18" rx="3" '
            f'fill="rgb({r},{g},{b})" fill-opacity="{alpha:.3f}">'
            f"<title>{_svg_text_escape(token)}: {impact:+.6f}</title></rect>"
        )
        parts.append(
            f'<text x="{bx + pad_x:.1f}" y="{by + 13:.1f}" font-size="{font_size}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'fill="{text_fill}">{_svg_text_escape(token)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def format_text_highlight_markdown(attrs: Sequence[Any]) -> str:
    """Classified text with each token shaded by impact, as inline SVG for Markdown reports."""
    svg = _svg_text_highlight(attrs)
    if not svg:
        return ""
    note = f"*{_HIGHLIGHT_LEGEND}*"
    truncated = _highlight_truncation_note(attrs)
    return "\n\n".join(p for p in (f'<figure class="dakma-text-highlight">{svg}</figure>', note, truncated) if p)


def format_text_highlight_html(attrs: Sequence[Any]) -> str:
    """Classified text as shaded ``<span>`` elements, for HTML reports."""
    rows = _pairs_token_impacts(attrs)[:_MAX_HIGHLIGHT_TOKENS]
    if not rows:
        return ""
    vmax = max(abs(v) for _, v in rows) or 1e-12
    spans: List[str] = []
    for token, impact in rows:
        (r, g, b), alpha = _token_shade(impact, vmax)
        fg = "#ffffff" if alpha > 0.5 else "#1a1a1a"
        spans.append(
            f'<span class="dakma-token" style="background-color: rgba({r},{g},{b},{alpha:.3f}); color: {fg};" '
            f'title="{html.escape(token, quote=True)}: {impact:+.6f}">{_html_esc(token)}</span>'
        )
    note = _highlight_truncation_note(attrs)
    note_html = _html_p_em(note.strip("*")) if note else ""
    return (
        f'<figure class="dakma-text-highlight"><p class="dakma-text-line">{" ".join(spans)}</p></figure>'
        f"{_html_p_em(_HIGHLIGHT_LEGEND)}{note_html}"
    )


def format_text_highlight_ansi(attrs: Sequence[Any]) -> str:
    """Classified text shaded with 24-bit ANSI background colours, for terminals."""
    rows = _pairs_token_impacts(attrs)[:_MAX_HIGHLIGHT_TOKENS]
    if not rows:
        return ""
    vmax = max(abs(v) for _, v in rows) or 1e-12
    out: List[str] = []
    for token, impact in rows:
        (r, g, b), alpha = _token_shade(impact, vmax)
        # Terminals have no alpha, so blend towards a white background ourselves.
        br, bg, bb = (round(255 + (c - 255) * alpha) for c in (r, g, b))
        fg = "255;255;255" if alpha > 0.5 else "26;26;26"
        out.append(f"\x1b[48;2;{br};{bg};{bb}m\x1b[38;2;{fg}m {token} \x1b[0m")
    return " ".join(out)


def _highlight_truncation_note(attrs: Sequence[Any]) -> str:
    if len(attrs) <= _MAX_HIGHLIGHT_TOKENS:
        return ""
    return f"*Showing the first {_MAX_HIGHLIGHT_TOKENS} of {len(attrs)} tokens.*"


def wrap_html_document(*, title: str, body_inner: str, subtitle: Optional[str] = None) -> str:
    """Wrap report body in a minimal printable HTML document."""
    sub = f'<p class="subtitle">{html.escape(subtitle, quote=False)}</p>' if subtitle else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title, quote=True)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem 2rem; color: #1a1a1a; line-height: 1.45; }}
    h1 {{ font-size: 1.35rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 0.5rem; }}
    h3 {{ font-size: 1.05rem; margin-top: 1.25rem; margin-bottom: 0.5rem; }}
    .subtitle {{ color: #555; margin-top: 0; font-size: 0.95rem; }}
    table.dakma-table {{ border-collapse: collapse; margin: 0.75rem 0 1.25rem; width: 100%; max-width: 56rem; font-size: 0.9rem; }}
    table.dakma-table th, table.dakma-table td {{ border: 1px solid #ccc; padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }}
    table.dakma-table th {{ background: #f4f4f4; }}
    section.entry {{ margin-bottom: 2rem; border-bottom: 1px solid #e0e0e0; padding-bottom: 1rem; }}
    figure.dakma-shap-chart {{ margin: 0.5rem 0 1rem; max-width: 56rem; }}
    figure.dakma-shap-chart svg {{ display: block; width: 100%; height: auto; max-width: 560px; }}
    figure.dakma-text-highlight {{ margin: 0.5rem 0 0.75rem; max-width: 56rem; }}
    p.dakma-text-line {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 2.1; margin: 0; }}
    span.dakma-token {{ padding: 0.15rem 0.3rem; border-radius: 3px; }}
    p.dakma-disclaimer {{ margin: 0 0 1.25rem; padding: 0.65rem 0.85rem; background: #fff8e6; border: 1px solid #e6d9b8; border-radius: 4px; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title, quote=False)}</h1>
  {sub}
  {body_inner}
</body>
</html>
"""


def _changelog_cell_html(value: Any) -> str:
    if value is None or value == "":
        return "— <em>not documented</em>"
    if isinstance(value, str):
        return _html_esc(value)
    if isinstance(value, list):
        lis: List[str] = []
        for item in value:
            if isinstance(item, dict):
                ver = item.get("version", item.get("v", ""))
                when = item.get("date", item.get("when", ""))
                notes = item.get("notes", item.get("text", item))
                lis.append(
                    f"<li><strong>{_html_esc(ver)}</strong> ({_html_esc(when)}): {_html_esc(notes)}</li>"
                )
            else:
                lis.append(f"<li>{_html_esc(item)}</li>")
        return "<ul>" + "".join(lis) + "</ul>" if lis else "— <em>not documented</em>"
    return _html_esc(value)


def format_governance_html(governance: Optional[Mapping[str, Any]] = None) -> str:
    g = dict(governance or {})
    std_keys = {k for k, _ in _GOVERNANCE_FIELD_ROWS}
    parts: List[str] = [
        _html_h2("Governance documentation"),
        _html_p_em(
            "Fields below are a documentation template only. "
            "Use DakmaClient.register_governance(...) to fill them. Empty items are flagged."
        ),
    ]
    rows: List[Sequence[Any]] = []
    for key, label in _GOVERNANCE_FIELD_ROWS:
        v = g.get(key)
        if key == "model_changelog" and v not in (None, ""):
            val: Any = _changelog_cell_html(v)
        else:
            val = "— <em>not documented</em>" if v is None or v == "" else _html_esc(v)
        rows.append((label, val))
    for k in sorted(x for x in g if x not in std_keys):
        val = g.get(k)
        rows.append((k, "— <em>not documented</em>" if val is None or val == "" else _html_esc(val)))
    parts.append(_html_table_two_col_doc_second_may_be_markup(rows))
    return "\n".join(parts)


def format_evaluation_block_html(
    evaluation: Optional[Mapping[str, Any]] = None,
    *,
    entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    ev = dict(evaluation or {})
    test_metrics: Dict[str, Any] = dict(ev.get("test_metrics") or {})
    n_train, n_test = ev.get("n_train"), ev.get("n_test")
    split_d = ev.get("split_description")
    cm = ev.get("confusion_matrix")

    parts: List[str] = [
        _html_h2("Evaluation and dataset"),
        _html_h3("Dataset and train/test split"),
        _html_table(
            ["Item", "Value"],
            [
                (
                    "Split / methodology description",
                    "— not provided" if not split_d else _html_esc(str(split_d)),
                ),
                ("Training set size (n_train)", n_train if n_train is not None else "— not provided"),
                ("Test / hold-out set size (n_test)", n_test if n_test is not None else "— not provided"),
            ],
        ),
        _html_h3("Test-set / hold-out metrics"),
        _html_p_em(
            "Report metrics on held-out data. Use DakmaClient.register_evaluation(test_metrics={...}, ...)."
        ),
    ]
    if test_metrics:
        tm_rows = [[k, _html_esc(v)] for k, v in sorted(test_metrics.items())]
    else:
        tm_rows = [("(none specified)", "— not provided")]
    parts.append(_html_table(["Metric (test/hold-out)", "Value"], tm_rows))

    want_display = [
        ("Precision (test)", "precision"),
        ("Recall (test)", "recall"),
        ("F1 (test)", "f1", "f1_score"),
        ("ROC-AUC (test)", "roc_auc", "roc_auc_score", "auc"),
        ("Accuracy (test)", "accuracy", "test_accuracy"),
    ]
    chk: List[Sequence[Any]] = []
    for spec in want_display:
        label, *keys = spec
        ok = _test_metric_keys_match(test_metrics, *keys) if test_metrics else False
        chk.append((label, "✓" if ok else "— not in test_metrics"))
    parts.append(_html_h3("Checklist: recommended test metrics"))
    parts.append(_html_table(["Item", "Status"], chk))

    parts.append(_html_h3("Confusion matrix (test / hold-out)"))
    if cm is not None and cm != "":
        try:
            arr = np.asarray(cm)
            if arr.shape == (2, 2):
                parts.append(
                    _html_table(
                        ["", "Pred 0", "Pred 1"],
                        [
                            ("Actual 0", int(arr[0, 0]), int(arr[0, 1])),
                            ("Actual 1", int(arr[1, 0]), int(arr[1, 1])),
                        ],
                    )
                )
            else:
                parts.append(f"<p>{_html_esc(cm)}</p>")
        except (TypeError, ValueError, IndexError):
            parts.append(f"<p>{_html_esc(cm)}</p>")
    else:
        parts.append(_html_p_em("Not provided — pass confusion_matrix= to register_evaluation."))

    parts.append(_html_h3("Gaps to address (automated notes)"))
    gap_bullets = _evaluation_gap_bullets(ev, test_metrics, entries=entries)
    if not gap_bullets:
        parts.append(
            _html_p_em(
                "No automated gap flags (still verify precision/recall/F1, lineage, and oversight manually)."
            )
        )
    else:
        parts.append(
            "<ul>"
            + "".join(f"<li>{html.escape(str(x), quote=False)}</li>" for x in gap_bullets)
            + "</ul>"
        )
    return "\n".join(parts)


def format_compliance_preamble_html(
    *,
    governance: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
    entries: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    g = format_governance_html(governance)
    e = format_evaluation_block_html(evaluation, entries=entries)
    return (
        f'<section class="dakma-compliance">{DOCUMENTATION_DISCLAIMER_HTML}\n'
        f"{g}\n{e}\n<hr></section>\n\n"
    )


def format_inference_result_html(
    result_dict: Dict[str, Any],
    *,
    include_compliance_sections: bool = True,
) -> str:
    """HTML version of :func:`format_inference_result_markdown`."""
    parts: List[str] = []
    explain0 = result_dict.get("explain") or {}
    if include_compliance_sections:
        g = explain0.get("governance") or {}
        e = explain0.get("evaluation") or {}
        parts.append(
            format_compliance_preamble_html(governance=g, evaluation=e, entries=None)
        )
    dec_out = result_dict.get("decision_output") or {}
    explain = result_dict.get("explain") or {}
    raw_decision = result_dict.get("decision")

    decision_rows = _decision_rows(dec_out)

    parts.append(_html_h3("Decision"))
    parts.append(
        _html_table(
            ["Field", "Value"],
            decision_rows,
        )
    )
    parts.append(_html_h3("Raw model output (unchanged)"))
    parts.append(_html_table(["decision"], [[_cell(raw_decision)]]))

    parts.append(_html_h3("Explainability"))
    meta_html = explain.get("metadata") or {}
    quality_html = explain.get("quality") or {}
    exp_rows = [
        ("audit_trail_id", explain.get("audit_trail_id")),
        ("model_version", explain.get("model_version")),
        (
            "attribution",
            explain.get("attribution_backend") or explain.get("method") or "shap",
        ),
        ("status", explain.get("status") or "—"),
        ("plain_language", explain.get("plain_language")),
        ("counterfactual", explain.get("counterfactual") or "—"),
    ]
    if meta_html.get("attribution_error"):
        exp_rows.append(("attribution_error", meta_html.get("attribution_error")))
    if quality_html.get("used_fallback"):
        exp_rows.append(("used_fallback", True))
    parts.append(_html_table(["Field", "Value"], exp_rows))

    flags = explain.get("regulation_flags") or []
    if flags:
        parts.append(_html_h3("Documentation status"))
        parts.append(_html_table(["Note"], [[f] for f in flags]))

    cu = explain.get("compute_usage")
    if cu and isinstance(cu, dict):
        parts.append(_format_compute_usage_html(cu))

    lh, ga, gf, la = _attribution_labels(explain)
    tokens = explain.get("text_attributions") or []
    if tokens:
        highlight = format_text_highlight_html(tokens)
        if highlight:
            parts.append(_html_h3("Highlighted text (this decision)"))
            parts.append(highlight)

    factors = explain.get("top_factors") or []
    if factors:
        parts.append(_html_h3(lh))
        parts.append(_html_shap_local_chart(factors, axis_label=la))

    fi_list = explain.get("feature_importance") or []
    if fi_list:
        parts.append(_html_h3("Feature importance (global)"))
        parts.append(_html_p_em(gf.strip("*")))
        parts.append(_html_shap_global_chart(fi_list, axis_label=ga))

    meta = explain.get("metadata") or {}
    if meta:
        parts.append(_html_h3("Metadata"))
        mr = [[k, _cell(v)] for k, v in sorted(meta.items()) if k != "feature_lineage"]
        parts.append(_html_table(["Key", "Value"], mr))
        fl = meta.get("feature_lineage") or {}
        if fl:
            parts.append(_html_h3("Feature lineage (snapshot)"))
            lr = [[k, _cell(v)] for k, v in sorted(fl.items())]
            parts.append(_html_table(["Feature", "Expression"], lr))

    return "\n".join(parts)


def format_audit_entry_html(entry: Dict[str, Any], index: int) -> str:
    """HTML for a single audit log entry."""
    etype = entry.get("type", "unknown")
    inner: List[str] = [f'<section class="entry" id="entry-{index}">']
    inner.append(f"<h2>{index}. {html.escape(str(etype), quote=False)}</h2>")

    if etype == "data_tracking":
        inner.append(
            _html_table(
                ["Field", "Value"],
                [
                    ("timestamp", entry.get("timestamp")),
                    ("function", entry.get("function")),
                ],
            )
        )
        inp = entry.get("input_schema") or {}
        out = entry.get("output_schema") or {}
        lin = entry.get("lineage_added") or {}
        if inp:
            inner.append(_html_h3("Input schema"))
            inner.append(_html_table(["Column", "dtype", "nulls"], _schema_to_rows(inp)))
        if out:
            inner.append(_html_h3("Output schema"))
            inner.append(_html_table(["Column", "dtype", "nulls"], _schema_to_rows(out)))
        if lin:
            inner.append(_html_h3("Feature lineage (new/updated columns)"))
            rows = [[k, _cell(v)] for k, v in sorted(lin.items())]
            inner.append(_html_table(["Feature", "Expression / source"], rows))
        inner.append("</section>")
        return "\n".join(inner)

    if etype == "training_tracking":
        inner.append(
            _html_table(
                ["Field", "Value"],
                [
                    ("timestamp", entry.get("timestamp")),
                    ("function", entry.get("function")),
                    ("model_class", entry.get("model_class")),
                    ("started_at", entry.get("started_at")),
                    ("finished_at", entry.get("finished_at")),
                ],
            )
        )
        params = entry.get("model_params") or {}
        if params:
            inner.append(_html_h3("Model parameters"))
            pr = [[k, _cell(v)] for k, v in sorted(params.items())]
            inner.append(_html_table(["Parameter", "Value"], pr))
        metrics = entry.get("metrics") or {}
        if metrics:
            inner.append(_html_h3("Metrics"))
            mr = [[k, _cell(v)] for k, v in sorted(metrics.items())]
            inner.append(_html_table(["Metric", "Value"], mr))
        cu = entry.get("compute_usage")
        if cu and isinstance(cu, dict):
            inner.append(_format_compute_usage_html(cu))
        inner.append("</section>")
        return "\n".join(inner)

    if etype == "monitoring":
        inner.append(
            _html_table(
                ["Field", "Value"],
                [("timestamp", entry.get("timestamp"))],
            )
        )
        payload = entry.get("payload") or {}
        inner.append(_html_h3("Monitoring summary"))
        acc = payload.get("accuracy")
        ppr = payload.get("positive_prediction_rate")
        drift = payload.get("drift_from_expected_rate")
        summary_rows = [
            ("accuracy", acc),
            ("positive_prediction_rate", ppr),
            ("drift_from_expected_rate", drift if drift is not None else "—"),
        ]
        inner.append(_html_table(["Metric", "Value"], summary_rows))
        groups = payload.get("group_positive_rates") or {}
        if groups:
            inner.append(_html_h3("Group positive prediction rates (bias / fairness signal)"))
            gr = [[g, _cell(r)] for g, r in sorted(groups.items())]
            inner.append(_html_table(["Group", "Mean predicted positive"], gr))
        inner.append("</section>")
        return "\n".join(inner)

    if etype == "inference_explain":
        inner.append(
            _html_table(
                ["Field", "Value"],
                [
                    ("timestamp", entry.get("timestamp")),
                    ("function", entry.get("function")),
                ],
            )
        )
        res = entry.get("result") or {}
        inner.append(format_inference_result_html(res, include_compliance_sections=False))
        inner.append("</section>")
        return "\n".join(inner)

    skip = {"type"}
    pairs = [(k, entry[k]) for k in sorted(entry.keys()) if k not in skip]
    inner.append(_html_table(["Field", "Value"], pairs))
    inner.append("</section>")
    return "\n".join(inner)


def format_audit_log_html(
    entries: Sequence[Dict[str, Any]],
    *,
    title: str = "Audit trail",
    governance: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
) -> str:
    """Full audit log as a complete HTML document (tables, print-friendly)."""
    preamble = format_compliance_preamble_html(
        governance=governance, evaluation=evaluation, entries=entries
    )
    blocks: List[str] = [preamble, _html_p_em(f"Total entries: {len(entries)}")]
    for i, entry in enumerate(entries, start=1):
        blocks.append(format_audit_entry_html(entry, i))
    body = "\n".join(blocks)
    return wrap_html_document(title=title, subtitle="Generated by dakma-sdk", body_inner=body)


def write_report_file(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write report text to ``path``; create parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return p


def write_audit_report(
    entries: Sequence[Dict[str, Any]],
    path: Union[str, Path],
    *,
    title: str = "Audit trail",
    format: Literal["markdown", "html"] = "markdown",
    governance: Optional[Mapping[str, Any]] = None,
    evaluation: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write the audit log to a downloadable ``.md`` or ``.html`` file."""
    if format == "markdown":
        body = format_audit_log_markdown(
            entries, title=title, governance=governance, evaluation=evaluation
        )
        return write_report_file(path, body)
    body = format_audit_log_html(
        entries, title=title, governance=governance, evaluation=evaluation
    )
    return write_report_file(path, body)


def write_inference_report(
    result_dict: Dict[str, Any],
    path: Union[str, Path],
    *,
    title: str = "Decision report",
    format: Literal["markdown", "html"] = "markdown",
) -> Path:
    """Write a single enriched decision to ``.md`` or ``.html``."""
    if format == "markdown":
        md = f"## {title}\n\n" + format_inference_result_markdown(result_dict)
        return write_report_file(path, md)
    inner = format_inference_result_html(result_dict)
    doc = wrap_html_document(title=title, subtitle="Single decision — dakma-sdk", body_inner=inner)
    return write_report_file(path, doc)


# Backward-compatible aliases (deprecated names).
format_governance_eu13_markdown = format_governance_markdown
format_governance_eu13_html = format_governance_html
