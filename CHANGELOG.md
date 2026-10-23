# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-22

### Fixed

- Local SHAP explanations now slice binary and multi-class attribution tensors to the scored class
  (aligned with `_extract_score` / last output index). This restores real SHAP factors for
  RandomForest / XGBoost multi-class models that previously fell through to the placeholder
  `feature_value × 0.01` path.

### Changed

- When SHAP fails and the coefficient / `value × 0.01` fallback is used, the explanation is marked
  `status="degraded"` with `metadata["attribution_error"]` prefixed by `fallback_used:` and
  `quality["used_fallback"]=true`. Decision and audit reports surface these fields.

### Added

- Regression tests for multi-class SHAP on Iris RandomForest and XGBoost
  (`tests/test_multiclass_shap.py`), plus unit coverage for class-axis slicing and fallback
  degradation.

## [1.0.0] - 2026-10-08

### Added

- `DakmaClient.explain_text` / `ModelHandle.explain_text`: Integrated Gradients attribution per
  token for text classification. Attributions are taken over word embeddings via a forward hook, so
  any model containing an `nn.Embedding` works without being restructured.
- `TokenAttribution` and `Explanation.text_attributions` hold the per-token impacts;
  `DecisionEvent.highlighted_text()` renders the text shaded by impact as ANSI, Markdown, or HTML.
- Markdown and HTML reports (including audit log entries) gain a **Highlighted text** section when a
  decision has token attributions.
- `dakma_sdk.text.token_attributions` and `find_embedding_layer` for computing token impacts outside
  the decorators.
- `examples/text_highlight_example.py`: runnable demo with ANSI terminal highlights and HTML reports
  under `examples/reports_text/`.

### Changed

- `reduce_attributions_to_features` accepts `reduce="tokens"` to sum a `(sequence, embedding)`
  attribution matrix over the embedding axis. Previously only the sequence axis was collapsed, which
  returned one impact per embedding dimension and discarded token names.
- Text attribution failures degrade the explanation instead of silently returning no factors: the
  reason is recorded in `explanation.metadata["attribution_error"]` and `status` becomes `degraded`.

### Notes

- First stable release (`Development Status :: 5 - Production/Stable`). The decision event and the
  six core objects are considered stable; breaking changes to them now require a major version.
- Upgrading from 0.5.0 needs no code changes.
- Reports and governance blocks remain **documentation aids**, not legal or regulatory certification.

## [0.5.0] - 2026-08-14

### Added

- Decision events as the SDK's fundamental object: `DecisionEvent` plus the six core types
  `Project`, `Model`, `Dataset`, `Decision`, `Explanation`, and `AuditEvent`.
- `DakmaClient.model(...)` returns a `ModelHandle` whose `explain()` /
  `explain_integrated_gradients()` record typed model provenance (framework, `model_hash`).
- `DakmaClient.dataset(...)` registers dataset identity; `data=` infers `row_count` and `schema_hash`.
- Each explained inference appends an `AuditEvent` of `type="decision"` to the audit log.

### Changed

- `DakmaClient.explain` and `explain_integrated_gradients` return a `DecisionEvent`, which keeps the
  `decision_output` and `explain` attribute names, so existing report and example code is unaffected.
- `init(project=...)` no longer requires `regulation` and `risk_level`.

## [0.1.0] - 2026-07-09

### Added

- Initial public release of `dakma-sdk` (import as `dakma` or `dakma_sdk`).
- `DakmaClient` decorators: `track_data`, `track_training`, `explain`, `explain_integrated_gradients`.
- Governance and evaluation documentation templates for audit/decision reports.
- Markdown and HTML audit reports with SHAP and Integrated Gradients attribution.
- Optional extras: `[ml]` (pandas, scikit-learn, XGBoost, SHAP), `[dl]` (PyTorch, Captum).
- Example scripts under `examples/` for XGBoost, SVM, and PyTorch MLP workflows.
- Unit and smoke tests (`pytest`).

### Changed

- Compliance language: documentation-aid disclaimers only; no certification checkmarks.
- Repository hygiene: `.gitignore`, `MANIFEST.in`; build artifacts no longer tracked.

### Notes

- This is an **alpha** release (`Development Status :: 3 - Alpha`).
- Reports and governance blocks are **documentation aids**, not legal or regulatory certification.

[1.1.0]: https://github.com/dakmaai/open_explainable_sdk/releases/tag/v1.1.0
[1.0.0]: https://github.com/dakmaai/open_explainable_sdk/releases/tag/v1.0.0
[0.5.0]: https://github.com/dakmaai/open_explainable_sdk/releases/tag/v0.5.0
[0.1.0]: https://github.com/dakmaai/open_explainable_sdk/releases/tag/v0.1.0
