from __future__ import annotations

import functools
import hashlib
import inspect
import pickle
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .compute_usage import ComputeUsageRecorder
from .models import (
    AuditEvent,
    Dataset,
    Decision,
    DecisionEvent,
    DecisionOutput,
    Explanation,
    FeatureImportance,
    Model,
    Project,
    TokenAttribution,
    TopFactor,
)


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def _safe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


_GOVERNANCE_FIELD_KEYS = (
    "intended_use",
    "known_limitations",
    "human_oversight",
    "data_provenance",
    "model_changelog",
)

_DOCUMENTATION_AID_NOTE = "Documentation aid only — not legal or compliance certification"


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_model_artifact(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    try:
        return _sha256_hex(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        try:
            return _sha256_hex(repr(artifact).encode("utf-8", errors="replace"))
        except Exception:
            return None


def _infer_framework(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    try:
        import torch.nn as nn  # type: ignore

        if isinstance(artifact, nn.Module):
            return "pytorch"
    except Exception:
        pass
    module = getattr(type(artifact), "__module__", "") or ""
    name = type(artifact).__name__.lower()
    if "xgboost" in module or name.startswith("xgb"):
        return "xgboost"
    if "sklearn" in module or "scikit" in module:
        return "sklearn"
    if module:
        return module.split(".", 1)[0]
    return None


class ModelHandle:
    """A registered model within a project, used to produce decision events.

    Created by :meth:`DakmaClient.model`. Decorating inference with :meth:`explain`
    records the model identity on every resulting :class:`~dakma_sdk.models.DecisionEvent`.
    """

    def __init__(
        self,
        client: "DakmaClient",
        *,
        name: str,
        version: str,
        framework: Optional[str] = None,
        model_hash: Optional[str] = None,
        artifact: Any = None,
        dataset: Optional[Dataset] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._artifact = artifact
        self._dataset = dataset
        self.spec = Model(
            id=name,
            version=version,
            framework=framework or _infer_framework(artifact),
            model_hash=model_hash or _hash_model_artifact(artifact),
            metadata=dict(metadata or {}),
        )

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def artifact(self) -> Any:
        return self._artifact

    @property
    def dataset(self) -> Optional[Dataset]:
        return self._dataset

    def bind(self, artifact: Any, *, rehash: bool = True) -> "ModelHandle":
        """Attach the runtime model object used for scoring and attribution."""
        self._artifact = artifact
        if self.spec.framework is None:
            self.spec.framework = _infer_framework(artifact)
        if rehash or not self.spec.model_hash:
            self.spec.model_hash = _hash_model_artifact(artifact)
        return self

    def use_dataset(self, dataset: Dataset) -> "ModelHandle":
        """Record ``dataset`` as the provenance for this model's decisions."""
        self._dataset = dataset
        return self

    def explain(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate inference so it returns a :class:`~dakma_sdk.models.DecisionEvent` (SHAP)."""
        return self._client._explain_decorator(
            self,
            backend="shap",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
        )

    def explain_integrated_gradients(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
        n_steps: int = 32,
        feature_names: Optional[Sequence[str]] = None,
        baseline: Any = None,
        target_class: Optional[int] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate inference so it returns a decision event explained with Integrated Gradients."""
        return self._client._explain_decorator(
            self,
            backend="integrated_gradients",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            feature_names=feature_names,
            baseline=baseline,
            target_class=target_class,
        )

    def explain_text(
        self,
        *,
        tokens: Any = None,
        tokenizer: Any = None,
        embedding_layer: Any = None,
        threshold: float = 0.5,
        top_k: int = 5,
        n_steps: int = 32,
        baseline: Any = None,
        target_class: Optional[int] = None,
        internal_batch_size: Optional[int] = 1,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate text inference so each token gets its own Integrated Gradients impact."""
        return self._client._explain_decorator(
            self,
            backend="integrated_gradients",
            input_kind="text",
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            baseline=baseline,
            target_class=target_class,
            tokens=tokens,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            internal_batch_size=internal_batch_size,
        )


class DakmaClient:
    def __init__(self, project: str, regulation: str = "", risk_level: str = ""):
        self.project = project
        self.regulation = regulation
        self.risk_level = risk_level
        self.project_info = Project(
            id=project,
            regulation=regulation or None,
            risk_level=risk_level or None,
        )
        self._audit_log: List[Dict[str, Any]] = []
        self._feature_lineage: Dict[str, str] = {}
        self._data_history: List[Dict[str, Any]] = []
        self._training_history: List[Dict[str, Any]] = []
        self._models: Dict[str, ModelHandle] = {}
        self._datasets: Dict[str, Dataset] = {}
        self._default_dataset: Optional[Dataset] = None
        self._shap_fi_model_id: Optional[int] = None
        self._shap_fi_rows: List[FeatureImportance] = []
        self._ig_fi_model_id: Optional[int] = None
        self._ig_fi_rows: List[FeatureImportance] = []
        self._governance: Dict[str, Any] = {}
        self._evaluation: Dict[str, Any] = {}

    def model(
        self,
        name: str,
        version: str = "0",
        *,
        framework: Optional[str] = None,
        model_hash: Optional[str] = None,
        artifact: Any = None,
        dataset: Optional[Union[str, Dataset]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelHandle:
        """Register a model and return a handle whose ``explain`` produces decision events.

        Pass ``artifact`` (the trained estimator) when the decorated function does not take
        the model as an argument, so attributions and the model hash can still be computed.
        """
        if isinstance(dataset, Dataset):
            bound_dataset: Optional[Dataset] = dataset
        elif isinstance(dataset, str):
            bound_dataset = self._datasets.get(dataset)
        else:
            bound_dataset = self._default_dataset

        handle = ModelHandle(
            self,
            name=name,
            version=version,
            framework=framework,
            model_hash=model_hash,
            artifact=artifact,
            dataset=bound_dataset,
            metadata=metadata,
        )
        self._models[f"{name}@{version}"] = handle
        return handle

    def dataset(
        self,
        id: str,
        version: str = "0",
        *,
        schema_hash: Optional[str] = None,
        row_count: Optional[int] = None,
        data: Any = None,
        set_default: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """Register dataset identity for provenance on later decision events.

        Passing ``data`` (a DataFrame, mapping, or array) fills in ``row_count`` and a
        column/dtype ``schema_hash`` when they are not given explicitly.
        """
        rows = row_count
        digest = schema_hash
        if data is not None:
            if rows is None:
                try:
                    rows = int(len(data))
                except Exception:
                    rows = None
            if digest is None:
                digest = self._schema_hash(data)
        ds = Dataset(
            id=id,
            version=version,
            schema_hash=digest,
            row_count=rows,
            metadata=dict(metadata or {}),
        )
        self._datasets[id] = ds
        if set_default:
            self._default_dataset = ds
        return ds

    def track_data(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_df = self._first_dataframe(args, kwargs)
            before_schema = self._schema_snapshot(input_df) if input_df is not None else {}

            result = fn(*args, **kwargs)

            output_df = result if self._is_dataframe(result) else None
            after_schema = self._schema_snapshot(output_df) if output_df is not None else {}
            lineage = self._extract_feature_lineage(fn)
            self._feature_lineage.update(lineage)

            entry = {
                "type": "data_tracking",
                "function": fn.__name__,
                "timestamp": self._utc_now(),
                "input_schema": before_schema,
                "output_schema": after_schema,
                "lineage_added": lineage,
            }
            self._data_history.append(entry)
            self._audit_log.append(entry)
            return result

        return wrapper

    def track_training(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = self._utc_now()
            with ComputeUsageRecorder() as rec:
                model = fn(*args, **kwargs)
            finished = self._utc_now()
            compute_usage = rec.usage

            model_params = self._extract_model_params(model)
            training_metrics = self._extract_training_metrics(model, args, kwargs)
            entry = {
                "type": "training_tracking",
                "function": fn.__name__,
                "timestamp": finished,
                "started_at": started,
                "finished_at": finished,
                "model_class": model.__class__.__name__,
                "model_params": model_params,
                "metrics": training_metrics,
                "compute_usage": compute_usage,
            }
            self._training_history.append(entry)
            self._audit_log.append(entry)
            return model

        return wrapper

    def register_governance(
        self,
        *,
        intended_use: Optional[str] = None,
        known_limitations: Optional[str] = None,
        human_oversight: Optional[str] = None,
        data_provenance: Optional[str] = None,
        model_changelog: Any = None,
        **extra: Any,
    ) -> None:
        """Record governance transparency text for audit and decision reports.

        This is a documentation template only; it does not certify regulatory compliance.
        Unknown keyword arguments are merged into the governance record. Empty values are
        ignored. Documentation appears in :meth:`write_audit_report` and in decision reports for
        subsequent explained inferences.
        """
        for key, value in (
            ("intended_use", intended_use),
            ("known_limitations", known_limitations),
            ("human_oversight", human_oversight),
            ("data_provenance", data_provenance),
            ("model_changelog", model_changelog),
        ):
            if value is not None and value != "":
                self._governance[key] = value
        for k, v in extra.items():
            if v is not None and v != "":
                self._governance[k] = v

    def register_evaluation(
        self,
        *,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        split_description: Optional[str] = None,
        test_metrics: Optional[Dict[str, Any]] = None,
        confusion_matrix: Any = None,
        **extra: Any,
    ) -> None:
        """Register hold-out / test evaluation and dataset split information for audit reports.

        Use ``test_metrics`` for test-set precision, recall, F1, ROC-AUC, etc. (not training
        fit scores). ``confusion_matrix`` may be a 2×2 list (e.g. ``[[tn, fp], [fn, tp]]``) or
        a mapping. Extra keys are merged into the evaluation record.
        """
        if n_train is not None:
            self._evaluation["n_train"] = n_train
        if n_test is not None:
            self._evaluation["n_test"] = n_test
        if split_description is not None and split_description != "":
            self._evaluation["split_description"] = split_description
        if test_metrics is not None:
            self._evaluation["test_metrics"] = {**self._evaluation.get("test_metrics", {}), **test_metrics}
        if confusion_matrix is not None:
            self._evaluation["confusion_matrix"] = confusion_matrix
        for k, v in extra.items():
            if v is not None and v != "":
                self._evaluation[k] = v

    def explain(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate inference so it returns a :class:`~dakma_sdk.models.DecisionEvent` (SHAP).

        Use :meth:`model` first when you want typed model and dataset provenance on the event.
        """
        return self._explain_decorator(
            None,
            backend="shap",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
        )

    def register_shap_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int = 500,
        top_k: Optional[int] = None,
    ) -> None:
        """Compute mean |SHAP| per feature over ``X_reference`` and attach it to later inference reports.

        Call after training with a background matrix (e.g. training or validation rows). Values are
        included on :class:`~dakma_sdk.models.DecisionEvent` only when the same ``model`` object is
        passed into the explained function. Requires optional dependencies ``shap`` and ``pandas``.
        """
        rows = self._global_shap_feature_importance(model, X_reference, max_samples=max_samples, top_k=top_k)
        self._shap_fi_rows = rows
        self._shap_fi_model_id = id(model) if model is not None and rows else None

    def explain_integrated_gradients(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
        n_steps: int = 32,
        feature_names: Optional[Sequence[str]] = None,
        baseline: Any = None,
        target_class: Optional[int] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Explain inference using Integrated Gradients (PyTorch ``nn.Module`` + tensor inputs).

        The wrapped function should return model outputs (logits or probabilities). Pass the trained
        ``model`` and input batch ``x`` (``torch.Tensor``) as arguments — convention:
        ``fn(model, x_tensor, ...)``. Requires ``torch``; optional ``captum`` for a fast IG
        implementation (``pip install dakma-sdk[dl]``).

        Global feature rankings can be precomputed with
        :meth:`register_integrated_gradients_feature_importance` using the same ``model`` object.
        """
        return self._explain_decorator(
            None,
            backend="integrated_gradients",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            feature_names=feature_names,
            baseline=baseline,
            target_class=target_class,
        )

    def explain_text(
        self,
        *,
        tokens: Any = None,
        tokenizer: Any = None,
        embedding_layer: Any = None,
        threshold: float = 0.5,
        top_k: int = 5,
        n_steps: int = 32,
        baseline: Any = None,
        target_class: Optional[int] = None,
        internal_batch_size: Optional[int] = 1,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Explain text classification with one Integrated Gradients impact per token.

        The wrapped function receives the model and a tensor of token ids shaped
        ``(1, seq_len)`` — convention: ``fn(model, ids)`` — and returns logits or
        probabilities. Attributions are taken over word embeddings, so the model only needs
        to contain an ``nn.Embedding`` (pass ``embedding_layer`` when it has several).

        Impacts explain the score the decision was made on — the last model output, as read by
        the threshold — so a positive impact pushed the text towards ``APPROVED``; pass
        ``target_class`` to attribute a different class instead.

        Supply display tokens with ``tokens`` (a list, or a callable taking the id tensor) or
        ``tokenizer`` (anything exposing ``convert_ids_to_tokens``). The resulting
        :class:`~dakma_sdk.models.DecisionEvent` carries per-token impacts on
        ``explanation.text_attributions``, renders them through
        :meth:`~dakma_sdk.models.DecisionEvent.highlighted_text`, and shades the text in
        reports. Requires ``torch``; ``captum`` is used when installed (``pip install dakma-sdk[dl]``).
        """
        return self._explain_decorator(
            None,
            backend="integrated_gradients",
            input_kind="text",
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            baseline=baseline,
            target_class=target_class,
            tokens=tokens,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            internal_batch_size=internal_batch_size,
        )

    def _explain_decorator(
        self,
        handle: Optional[ModelHandle],
        *,
        backend: str,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
        n_steps: int = 32,
        feature_names: Optional[Sequence[str]] = None,
        baseline: Any = None,
        target_class: Optional[int] = None,
        input_kind: str = "tabular",
        tokens: Any = None,
        tokenizer: Any = None,
        embedding_layer: Any = None,
        internal_batch_size: Optional[int] = 1,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., DecisionEvent]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> DecisionEvent:
                bound = handle.artifact if handle is not None else None
                token_attrs: List[TokenAttribution] = []
                attribution_error: Optional[str] = None
                with ComputeUsageRecorder() as rec:
                    raw_decision = fn(*args, **kwargs)

                    if input_kind == "text":
                        model = bound if bound is not None else self._first_pytorch_module(args, kwargs)
                        token_attrs, attribution_error = self._token_attributions(
                            model,
                            self._first_token_id_tensor(args, kwargs),
                            tokens=tokens,
                            tokenizer=tokenizer,
                            embedding_layer=embedding_layer,
                            n_steps=n_steps,
                            baseline=baseline,
                            target_class=(
                                target_class
                                if target_class is not None
                                else self._score_class_index(raw_decision)
                            ),
                            internal_batch_size=internal_batch_size,
                        )
                        top_factors = self._token_top_factors(token_attrs, top_k=top_k)
                        cached_id, cached_rows = None, []
                    elif backend == "integrated_gradients":
                        model = bound if bound is not None else self._first_pytorch_module(args, kwargs)
                        top_factors = self._top_factors_integrated_gradients(
                            model,
                            self._first_torch_tensor(args, kwargs),
                            feature_names=feature_names,
                            top_k=top_k,
                            n_steps=n_steps,
                            baseline=baseline,
                            target_class=target_class,
                        )
                        cached_id, cached_rows = self._ig_fi_model_id, self._ig_fi_rows
                    else:
                        model = bound if bound is not None else self._first_model(args, kwargs)
                        row = self._first_dataframe_or_mapping(args, kwargs)
                        top_factors, attribution_error = self._top_factors(
                            model,
                            row,
                            top_k=top_k,
                            class_index=self._score_class_index(raw_decision),
                        )
                        cached_id, cached_rows = self._shap_fi_model_id, self._shap_fi_rows

                    fi_rows: List[FeatureImportance] = []
                    if cached_id is not None and model is not None and id(model) == cached_id:
                        fi_rows = list(cached_rows)

                    decision = self._to_decision(self._extract_score(raw_decision), threshold)
                    counterfactual_text = self._counterfactual(top_factors) if counterfactual else None

                event = self._build_decision_event(
                    handle=handle,
                    model=model,
                    decision=decision,
                    factors=top_factors,
                    method=backend,
                    counterfactual_text=counterfactual_text,
                    feature_importance=fi_rows,
                    compute_usage=rec.usage,
                    raw_output=raw_decision,
                    input_kind=input_kind,
                    text_attributions=token_attrs,
                    attribution_error=attribution_error,
                )
                self._record_decision_event(event, function_name=fn.__name__)
                return event

            return wrapper

        return decorator

    def _build_decision_event(
        self,
        *,
        handle: Optional[ModelHandle],
        model: Any,
        decision: Decision,
        factors: List[TopFactor],
        method: str,
        counterfactual_text: Optional[str],
        feature_importance: List[FeatureImportance],
        compute_usage: Optional[Dict[str, Any]],
        raw_output: Any,
        input_kind: str = "tabular",
        text_attributions: Optional[List[TokenAttribution]] = None,
        attribution_error: Optional[str] = None,
    ) -> DecisionEvent:
        audit_id = self._audit_trail_id()
        governance = dict(self._governance) if self._governance else None
        evaluation = dict(self._evaluation) if self._evaluation else None

        if handle is not None:
            spec: Model = handle.spec
            model_version = f"{spec.id}:{spec.version}"
            dataset = handle.dataset or self._default_dataset
        else:
            spec = Model(
                id=self.project,
                version=self._model_version(model),
                framework=_infer_framework(model),
                model_hash=_hash_model_artifact(model),
            )
            model_version = spec.version
            dataset = self._default_dataset

        tokens = list(text_attributions or [])
        metadata: Dict[str, Any] = {
            "project": self.project,
            "regulation": self.regulation,
            "risk_level": self.risk_level,
            "feature_lineage": dict(self._feature_lineage),
        }
        if input_kind != "tabular":
            metadata["input_kind"] = input_kind
        if attribution_error:
            metadata["attribution_error"] = attribution_error

        explanation = Explanation(
            method=method,
            status="success" if factors and not attribution_error else "degraded",
            factors=factors,
            quality={
                "factor_count": len(factors),
                "has_global_importance": bool(feature_importance),
                "method": method,
                "used_fallback": bool(
                    attribution_error and str(attribution_error).startswith("fallback_used:")
                ),
                **({"token_count": len(tokens)} if tokens else {}),
            },
            plain_language=self._plain_language(decision, factors, input_kind=input_kind),
            counterfactual=counterfactual_text,
            regulation_flags=self._regulation_flags(),
            feature_importance=feature_importance,
            compute_usage=compute_usage,
            metadata=metadata,
            audit_trail_id=audit_id,
            model_version=model_version,
            governance=governance,
            evaluation=evaluation,
            text_attributions=tokens,
        )
        return DecisionEvent(
            decision=decision,
            explanation=explanation,
            audit_id=audit_id,
            model=spec,
            dataset=dataset,
            project=self.project_info,
            governance=governance,
            evaluation=evaluation,
            raw_output=raw_output,
        )

    def _record_decision_event(self, event: DecisionEvent, *, function_name: str) -> AuditEvent:
        timestamp = self._utc_now()
        audit_event = event.to_audit_event(timestamp=timestamp)
        self._audit_log.append(
            {
                "type": "inference_explain",
                "timestamp": timestamp,
                "function": function_name,
                "result": event.to_report_dict(),
                "audit_event": audit_event.as_dict(),
            }
        )
        return audit_event

    def _schema_hash(self, data: Any) -> Optional[str]:
        pd = _safe_import_pandas()
        try:
            if pd is not None and isinstance(data, pd.DataFrame):
                return _sha256_hex("|".join(f"{c}:{data[c].dtype}" for c in data.columns).encode("utf-8"))
            if isinstance(data, dict):
                payload = "|".join(f"{k}:{type(v).__name__}" for k, v in sorted(data.items()))
                return _sha256_hex(payload.encode("utf-8"))
            arr = np.asarray(data)
            return _sha256_hex(f"ndarray:{arr.shape}:{arr.dtype}".encode("utf-8"))
        except Exception:
            return None

    def register_integrated_gradients_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int = 200,
        top_k: Optional[int] = None,
        n_steps: int = 32,
        baseline: Any = None,
        target_class: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Mean |Integrated Gradients| per feature over a reference tensor (tabular / channel-wise).

        Same pattern as :meth:`register_shap_feature_importance`: values appear on reports when the
        same ``model`` instance is used with :meth:`explain_integrated_gradients`.
        """
        rows = self._global_integrated_gradients_importance(
            model,
            X_reference,
            max_samples=max_samples,
            top_k=top_k,
            n_steps=n_steps,
            baseline=baseline,
            target_class=target_class,
            feature_names=feature_names,
        )
        self._ig_fi_rows = rows
        self._ig_fi_model_id = id(model) if model is not None and rows else None

    def monitor(
        self,
        *,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        protected_feature: Optional[Sequence[Any]] = None,
        expected_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        accuracy = float((y_true_arr == y_pred_arr).mean())
        pred_positive_rate = float(y_pred_arr.mean()) if len(y_pred_arr) else 0.0
        drift = None
        if expected_rate is not None:
            drift = abs(pred_positive_rate - expected_rate)

        bias_metrics = {}
        if protected_feature is not None and len(protected_feature) == len(y_pred_arr):
            groups = {}
            for group, pred in zip(protected_feature, y_pred_arr):
                groups.setdefault(str(group), []).append(int(pred))
            bias_metrics = {g: float(np.mean(v)) for g, v in groups.items()}

        payload = {
            "accuracy": accuracy,
            "positive_prediction_rate": pred_positive_rate,
            "drift_from_expected_rate": drift,
            "group_positive_rates": bias_metrics,
        }
        self._audit_log.append({"type": "monitoring", "timestamp": self._utc_now(), "payload": payload})
        return payload

    def export_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def format_audit_log_markdown(self, *, title: str = "Audit trail") -> str:
        """Return the full audit log as readable Markdown tables (per entry type)."""
        from .audit_format import format_audit_log_markdown

        return format_audit_log_markdown(
            self._audit_log,
            title=title,
            governance=self._governance,
            evaluation=self._evaluation,
        )

    def write_audit_report(
        self,
        path: Union[str, Path],
        *,
        title: str = "Audit trail",
        format: Literal["markdown", "html"] = "markdown",
    ) -> Path:
        """Write the full audit log to a downloadable ``.md`` or ``.html`` file (UTF-8)."""
        from .audit_format import write_audit_report as _write_audit_report

        return _write_audit_report(
            self._audit_log,
            path,
            title=title,
            format=format,
            governance=self._governance,
            evaluation=self._evaluation,
        )

    def last_data_report(self) -> Optional[Dict[str, Any]]:
        return self._data_history[-1] if self._data_history else None

    def last_training_report(self) -> Optional[Dict[str, Any]]:
        return self._training_history[-1] if self._training_history else None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_dataframe(value: Any) -> bool:
        pd = _safe_import_pandas()
        return pd is not None and isinstance(value, pd.DataFrame)

    def _first_dataframe(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        pd = _safe_import_pandas()
        if pd is None:
            return None
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                return arg
        for val in kwargs.values():
            if isinstance(val, pd.DataFrame):
                return val
        return None

    def _schema_snapshot(self, df: Any) -> Dict[str, Dict[str, Any]]:
        if df is None:
            return {}
        return {
            col: {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
            }
            for col in df.columns
        }

    def _extract_feature_lineage(self, fn: Callable[..., Any]) -> Dict[str, str]:
        source = inspect.getsource(fn)
        pattern = r"""df\["([^"]+)"\]\s*=\s*(.+)"""
        found = re.findall(pattern, source)
        lineage = {}
        for feature, expression in found:
            lineage[feature] = expression.strip()
        return lineage

    @staticmethod
    def _extract_model_params(model: Any) -> Dict[str, Any]:
        if hasattr(model, "get_params"):
            try:
                return model.get_params()
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_training_metrics(model: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        if hasattr(model, "score"):
            X_train = kwargs.get("X_train")
            if X_train is None and len(args) > 0:
                X_train = args[0]
            y_train = kwargs.get("y_train")
            if y_train is None and len(args) > 1:
                y_train = args[1]
            if X_train is not None and y_train is not None:
                try:
                    metrics["training_score"] = float(model.score(X_train, y_train))
                except Exception:
                    pass
        return metrics

    @staticmethod
    def _first_model(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        for arg in args:
            if hasattr(arg, "predict") or hasattr(arg, "predict_proba") or hasattr(arg, "get_params"):
                return arg
        for value in kwargs.values():
            if hasattr(value, "predict") or hasattr(value, "predict_proba") or hasattr(value, "get_params"):
                return value
        return None

    @staticmethod
    def _first_pytorch_module(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            import torch.nn as nn  # type: ignore
        except Exception:
            return None
        for arg in args:
            if isinstance(arg, nn.Module):
                return arg
        for value in kwargs.values():
            if isinstance(value, nn.Module):
                return value
        return None

    @staticmethod
    def _first_torch_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            import torch  # type: ignore
        except Exception:
            return None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _first_dataframe_or_mapping(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        df = self._first_dataframe(args, kwargs)
        if df is not None:
            return df
        for arg in args:
            if isinstance(arg, dict):
                return arg
        for value in kwargs.values():
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _extract_score(raw_decision: Any) -> float:
        try:
            import torch  # type: ignore

            if isinstance(raw_decision, torch.Tensor):
                raw_decision = raw_decision.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(raw_decision)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1])
        if arr.ndim >= 2:
            return float(arr[0, -1])
        return 0.0

    @staticmethod
    def _score_class_index(raw_decision: Any) -> Optional[int]:
        """Output index that :meth:`_extract_score` read, so attributions explain that score."""
        try:
            import torch  # type: ignore

            if isinstance(raw_decision, torch.Tensor):
                raw_decision = raw_decision.detach().cpu().numpy()
        except Exception:
            pass
        try:
            arr = np.asarray(raw_decision)
        except Exception:
            return None
        if arr.ndim == 0 or arr.shape[-1] == 0:
            return None
        return int(arr.shape[-1] - 1)

    @staticmethod
    def _to_decision_output(score: float, threshold: float) -> DecisionOutput:
        if score >= threshold:
            return DecisionOutput(label="APPROVED", score=score, threshold=threshold)
        return DecisionOutput(label="DECLINED", score=score, threshold=threshold)

    @staticmethod
    def _to_decision(score: float, threshold: float) -> Decision:
        value = "APPROVED" if score >= threshold else "DECLINED"
        return Decision(value=value, score=score, threshold=threshold)

    def _top_factors(
        self,
        model: Any,
        row: Any,
        top_k: int = 3,
        *,
        class_index: Optional[int] = None,
    ) -> Tuple[List[TopFactor], Optional[str]]:
        """Return local factors and an optional attribution error.

        On SHAP failure, factors may still come from :meth:`_top_factors_fallback`; in that case
        the error string starts with ``fallback_used:`` so reports can mark the explanation
        ``degraded``.
        """
        factors, shap_error = self._top_factors_shap(model, row, top_k=top_k, class_index=class_index)
        if factors:
            return factors[:top_k], None
        fallback = self._top_factors_fallback(model, row)
        if fallback:
            detail = shap_error or "SHAP produced no local factors"
            return fallback[:top_k], f"fallback_used: {detail}"
        return [], shap_error or "no local factors available"

    def _reference_dataframe(self, X_reference: Any) -> Optional[Any]:
        pd = _safe_import_pandas()
        if pd is None:
            return None
        if isinstance(X_reference, pd.DataFrame):
            return X_reference
        if isinstance(X_reference, dict):
            return pd.DataFrame([X_reference])
        if isinstance(X_reference, np.ndarray):
            arr = X_reference
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                return None
            cols = [f"f{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
        return None

    def _global_shap_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int,
        top_k: Optional[int],
    ) -> List[FeatureImportance]:
        shap = _safe_import_shap()
        row_df = self._reference_dataframe(X_reference)
        if shap is None or model is None or row_df is None or len(row_df) == 0:
            return []
        try:
            if len(row_df) > max_samples:
                row_df = row_df.sample(n=max_samples, random_state=0)
            # Pass the same reference rows as the masker so sklearn / non-tree models work with SHAP.
            explanation = shap.Explainer(model, row_df)(row_df)
            vals = np.asarray(explanation.values, dtype=float)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
            if vals.ndim == 3:
                mean_abs = np.mean(np.abs(vals), axis=(0, 2))
            elif vals.ndim == 2:
                mean_abs = np.mean(np.abs(vals), axis=0)
            else:
                return []
            features = list(row_df.columns)
            if mean_abs.shape[0] != len(features):
                return []
            pairs = sorted(zip(features, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[: int(top_k)]
            return [FeatureImportance(name=str(n), mean_abs_shap=float(v), source="shap") for n, v in pairs]
        except Exception:
            return []

    def _global_integrated_gradients_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int,
        top_k: Optional[int],
        n_steps: int,
        baseline: Any,
        target_class: Optional[int],
        feature_names: Optional[Sequence[str]],
    ) -> List[FeatureImportance]:
        try:
            import torch  # type: ignore

            from .integrated_gradients import (
                compute_integrated_gradients_attributions,
                reduce_attributions_to_features,
            )
        except Exception:
            return []

        if model is None:
            return []
        try:
            if isinstance(X_reference, torch.Tensor):
                x_all = X_reference.float()
            else:
                x_all = torch.as_tensor(np.asarray(X_reference), dtype=torch.float32)
            if x_all.dim() < 2:
                return []
            n = min(int(x_all.shape[0]), int(max_samples))
            if n <= 0:
                return []
            mean_abs: Optional[np.ndarray] = None
            names: Optional[List[str]] = None
            for i in range(n):
                row = x_all[i : i + 1]
                attr = compute_integrated_gradients_attributions(
                    model,
                    row,
                    baseline=baseline,
                    n_steps=n_steps,
                    target_class=target_class,
                )
                nms, _, impacts = reduce_attributions_to_features(attr, feature_names=feature_names)
                contrib = np.abs(np.asarray(impacts, dtype=float))
                if mean_abs is None:
                    mean_abs = contrib
                    names = nms
                elif contrib.shape == mean_abs.shape:
                    mean_abs = mean_abs + contrib
                else:
                    return []
            assert mean_abs is not None and names is not None
            mean_abs = mean_abs / float(n)
            pairs = sorted(zip(names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[: int(top_k)]
            return [
                FeatureImportance(name=str(a), mean_abs_shap=float(b), source="integrated_gradients")
                for a, b in pairs
            ]
        except Exception:
            return []

    def _top_factors_integrated_gradients(
        self,
        model: Any,
        x_tensor: Any,
        *,
        feature_names: Optional[Sequence[str]],
        top_k: int,
        n_steps: int,
        baseline: Any,
        target_class: Optional[int],
    ) -> List[TopFactor]:
        if model is None or x_tensor is None:
            return []
        try:
            from .integrated_gradients import (
                compute_integrated_gradients_attributions,
                input_values_for_display,
                reduce_attributions_to_features,
            )

            attr = compute_integrated_gradients_attributions(
                model,
                x_tensor,
                baseline=baseline,
                n_steps=n_steps,
                target_class=target_class,
            )
            names, _, impacts = reduce_attributions_to_features(attr, feature_names=feature_names)
            vals = input_values_for_display(x_tensor, len(names))
            sortable = list(zip(names, vals, impacts))
            sortable.sort(key=lambda x: abs(x[2]), reverse=True)
            out: List[TopFactor] = []
            for name, val, impact in sortable[:top_k]:
                out.append(
                    TopFactor(
                        name=name,
                        value=float(val),
                        impact=float(impact),
                        direction="up" if impact >= 0 else "down",
                    )
                )
            return out
        except Exception:
            return []

    def _token_attributions(
        self,
        model: Any,
        ids: Any,
        *,
        tokens: Any,
        tokenizer: Any,
        embedding_layer: Any,
        n_steps: int,
        baseline: Any,
        target_class: Optional[int],
        internal_batch_size: Optional[int],
    ) -> Tuple[List[TokenAttribution], Optional[str]]:
        """Per-token impacts, plus the reason attribution degraded (``None`` on success)."""
        if model is None:
            return [], "no torch.nn.Module found in the decorated call"
        if ids is None:
            return [], "no token id tensor found in the decorated call"
        try:
            from .text import token_attributions

            return (
                token_attributions(
                    model,
                    ids,
                    tokens=tokens,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    n_steps=n_steps,
                    baseline=baseline,
                    target_class=target_class,
                    internal_batch_size=internal_batch_size,
                ),
                None,
            )
        except Exception as exc:  # keep inference usable; the event records why it degraded
            return [], f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _token_top_factors(attributions: List[TokenAttribution], *, top_k: int) -> List[TopFactor]:
        ranked = sorted(attributions, key=lambda t: abs(t.impact), reverse=True)[:top_k]
        return [
            TopFactor(name=t.token, value=t.index, impact=t.impact, direction=t.direction)
            for t in ranked
        ]

    @staticmethod
    def _first_token_id_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            import torch  # type: ignore
        except Exception:
            return None
        fallback = None
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor):
                if not torch.is_floating_point(value):
                    return value
                if fallback is None:
                    fallback = value
        return fallback

    @staticmethod
    def _local_shap_vector(
        shap_values: Any,
        *,
        n_features: int,
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """Reduce one-row SHAP values to a length-``n_features`` vector for the scored class.

        Handles common Explainer layouts:

        - ``(n_features,)`` — already a single-output / selected-class vector
        - ``(n_features, n_classes)`` — binary / multi-class (slice by ``class_index``)
        - ``(n_classes, n_features)`` — less common class-major layout
        """
        vals = np.asarray(shap_values, dtype=float)
        if vals.ndim == 1:
            if int(vals.shape[0]) != int(n_features):
                raise ValueError(
                    f"SHAP vector length {vals.shape[0]} does not match n_features={n_features}"
                )
            return vals
        if vals.ndim == 2:
            if int(vals.shape[0]) == int(n_features):
                n_classes = int(vals.shape[1])
                idx = n_classes - 1 if class_index is None else int(class_index)
                idx = max(0, min(idx, n_classes - 1))
                return vals[:, idx]
            if int(vals.shape[1]) == int(n_features):
                n_classes = int(vals.shape[0])
                idx = n_classes - 1 if class_index is None else int(class_index)
                idx = max(0, min(idx, n_classes - 1))
                return vals[idx, :]
        raise ValueError(
            f"Unsupported SHAP value shape {getattr(vals, 'shape', None)} for n_features={n_features}"
        )

    def _top_factors_shap(
        self,
        model: Any,
        row: Any,
        *,
        top_k: int = 3,
        class_index: Optional[int] = None,
    ) -> Tuple[List[TopFactor], Optional[str]]:
        shap = _safe_import_shap()
        pd = _safe_import_pandas()
        if shap is None:
            return [], "shap is not installed"
        if model is None or row is None or pd is None:
            return [], "model, row, or pandas unavailable for SHAP"

        try:
            row_df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
            explainer = shap.Explainer(model)
            values = explainer(row_df)
            features = list(row_df.columns)
            shap_vals = self._local_shap_vector(
                values.values[0],
                n_features=len(features),
                class_index=class_index,
            )
            sortable = list(zip(features, row_df.iloc[0].tolist(), shap_vals.tolist()))
            sortable.sort(key=lambda x: abs(float(x[2])), reverse=True)
            result: List[TopFactor] = []
            for name, val, impact in sortable[:top_k]:
                impact_f = float(impact)
                direction = "up" if impact_f >= 0 else "down"
                result.append(TopFactor(name=name, value=val, impact=impact_f, direction=direction))
            if not result:
                return [], "SHAP returned an empty factor list"
            return result, None
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

    def _top_factors_fallback(self, model: Any, row: Any) -> List[TopFactor]:
        pd = _safe_import_pandas()
        if row is None:
            return []
        if pd is not None and isinstance(row, pd.DataFrame):
            features = list(row.columns)
            values = row.iloc[0].tolist()
        elif isinstance(row, dict):
            features = list(row.keys())
            values = list(row.values())
        else:
            return []

        impacts = np.zeros(len(features), dtype=float)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).reshape(-1)
            if len(coef) == len(features):
                impacts = coef * np.asarray(values, dtype=float)
        else:
            impacts = np.asarray(values, dtype=float) * 0.01

        sortable = list(zip(features, values, impacts))
        sortable.sort(key=lambda x: abs(x[2]), reverse=True)
        top = []
        for name, val, impact in sortable[:3]:
            top.append(TopFactor(name=name, value=val, impact=float(impact), direction="up" if impact >= 0 else "down"))
        return top

    @staticmethod
    def _counterfactual(factors: List[TopFactor]) -> Optional[str]:
        if not factors:
            return None
        dominant = factors[0]
        if dominant.impact < 0:
            try:
                target = float(dominant.value) * 0.7
                return f"Would approve if {dominant.name} < {target:.2f}"
            except Exception:
                return f"Would approve if {dominant.name} is improved"
        return f"Would decline if {dominant.name} worsens further"

    def _model_version(self, model: Any) -> str:
        if model is None:
            return f"{self.project} / unknown-model"
        try:
            import torch.nn as nn  # type: ignore

            if isinstance(model, nn.Module):
                return f"{self.project} / torch-{model.__class__.__name__.lower()}"
        except Exception:
            pass
        params = self._extract_model_params(model)
        n_estimators = params.get("n_estimators")
        max_depth = params.get("max_depth")
        suffix = f"{model.__class__.__name__.lower()}"
        if n_estimators is not None and max_depth is not None:
            suffix = f"xgb-{n_estimators}-d{max_depth}"
        return f"{self.project} / {suffix}"

    def _plain_language(
        self,
        decision: Union[Decision, DecisionOutput],
        factors: List[TopFactor],
        *,
        input_kind: str = "tabular",
    ) -> str:
        if not factors:
            return f"{decision.label.title()}. Decision generated by model score and threshold."
        reason = factors[0]
        if input_kind == "text":
            return (
                f"{decision.label.title()}. Main driver: the token \"{reason.name}\" "
                f"which moved the score {'down' if reason.impact < 0 else 'up'}."
            )
        direction = "high" if isinstance(reason.value, (int, float)) and float(reason.value) > 0 else "low"
        return (
            f"{decision.label.title()}. Main reason: {direction} {reason.name.replace('_', ' ')} "
            f"which moved the score {'down' if reason.impact < 0 else 'up'}."
        )

    def _regulation_flags(self) -> List[str]:
        """Human-readable documentation status notes (not compliance certification)."""
        populated = sum(
            1 for key in _GOVERNANCE_FIELD_KEYS if self._governance.get(key) not in (None, "")
        )
        total = len(_GOVERNANCE_FIELD_KEYS)
        notes = [f"Governance documentation fields: {populated}/{total} populated"]
        if self.regulation:
            notes.append(f"Policy context: {self.regulation}")
        notes.append(_DOCUMENTATION_AID_NOTE)
        return notes

    @staticmethod
    def _audit_trail_id() -> str:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        token = uuid.uuid4().hex[:7]
        return f"ax-{date}-{token}"
