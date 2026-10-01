"""Laya decision-model adapter for the V0.8 trading meta-policy.

The runtime dependency is intentionally lazy. Offline unit tests, dataset
construction, and the quantitative stack do not need Laya installed. A real
checkpoint is loaded only when inference is explicitly requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


LAYA_TRADING_QUESTIONS: dict[str, dict[str, Any]] = {
    "action": {
        "type": "choice",
        "instructions": (
            "Given the supplied historical market state, decide whether this setup "
            "supports the proposed short-horizon direction. Choose WAIT when the "
            "evidence is weak, conflicting, unusually risky, or the edge is not clear."
        ),
        "criteria": {
            "LONG": "The state supports a short-horizon long setup.",
            "WAIT": "The state does not justify a directional trade.",
            "SHORT": "The state supports a short-horizon short setup.",
        },
    },
    "quality": {
        "type": "choice",
        "instructions": "Rate the setup quality using only the supplied state.",
        "criteria": {
            "POOR": "No robust edge or conditions are materially adverse.",
            "FAIR": "Some edge exists but the setup is marginal or mixed.",
            "GOOD": "The setup has a clear edge with acceptable context.",
            "EXCELLENT": "The setup has unusually strong, aligned evidence.",
        },
    },
    "risk_concern": {
        "type": "noul",
        "instructions": (
            "Is there a material contextual risk or uncertainty that should cause "
            "the trading system to reject this otherwise quantitative candidate?"
        ),
    },
}


class LayaAgentProtocol(Protocol):
    def predict(self, state: Any, questions: Mapping[str, Any], **kwargs: Any) -> dict: ...

    def predict_batch(
        self,
        states: Sequence[Any],
        questions: Mapping[str, Any],
        **kwargs: Any,
    ) -> list[dict]: ...


@dataclass(frozen=True)
class LayaDecision:
    action: str
    action_probabilities: dict[str, float]
    action_answer_confidence: float
    quality: str
    quality_probabilities: dict[str, float]
    quality_answer_confidence: float
    risk_concern_probability: float
    model: str | None = None

    @property
    def risk_concern(self) -> bool:
        return self.risk_concern_probability >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _answer_map(result: Mapping[str, Any]) -> Mapping[str, Any]:
    answers = result.get("answers", result)
    if not isinstance(answers, Mapping):
        raise ValueError("Laya result does not contain an answer mapping")
    return answers


def _choice_answer(answers: Mapping[str, Any], key: str) -> tuple[str, dict[str, float], float]:
    payload = answers.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Laya result missing {key!r} answer")
    choice = str(payload.get("choice", "")).upper()
    raw_probabilities = payload.get("probabilities", {})
    probabilities = {
        str(label).upper(): float(value)
        for label, value in raw_probabilities.items()
    } if isinstance(raw_probabilities, Mapping) else {}
    answer_confidence = payload.get("answer_confidence")
    if answer_confidence is None:
        answer_confidence = max(probabilities.values(), default=0.0)
    return choice, probabilities, float(answer_confidence)


def normalize_laya_result(result: Mapping[str, Any]) -> LayaDecision:
    """Normalize the current Laya response schema into one stable project type."""

    answers = _answer_map(result)
    action, action_probabilities, action_confidence = _choice_answer(answers, "action")
    quality, quality_probabilities, quality_confidence = _choice_answer(answers, "quality")

    risk_payload = answers.get("risk_concern")
    if not isinstance(risk_payload, Mapping):
        # Compatibility with the v0.3 prototype question name.
        risk_payload = answers.get("high_risk")
    if not isinstance(risk_payload, Mapping):
        raise ValueError("Laya result missing risk_concern answer")
    risk_probability = float(risk_payload.get("noul", 0.5))

    model = result.get("model")
    if model is None and isinstance(result.get("routing"), Mapping):
        model = result["routing"].get("repo") or result["routing"].get("model")

    return LayaDecision(
        action=action,
        action_probabilities=action_probabilities,
        action_answer_confidence=action_confidence,
        quality=quality,
        quality_probabilities=quality_probabilities,
        quality_answer_confidence=quality_confidence,
        risk_concern_probability=risk_probability,
        model=str(model) if model is not None else None,
    )


class LayaEngine:
    """Lazy Laya checkpoint wrapper with batch inference support.

    ``agent`` may be injected for deterministic tests or an externally managed
    runtime. If omitted, ``laya.load`` is imported and called only on first use.
    """

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        *,
        agent: LayaAgentProtocol | None = None,
        device: str | None = None,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> None:
        self.model_id = model_id
        self._agent = agent
        self.device = device
        self.max_len = max_len
        self.head_max_len = head_max_len

    def _load(self) -> LayaAgentProtocol:
        if self._agent is not None:
            return self._agent
        try:
            import laya  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "Laya runtime is not installed. Install the optional `laya` package "
                "and the appropriate PyTorch build before requesting V0.8 inference."
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self._agent = laya.load(self.model_id, **kwargs)
        return self._agent

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.max_len is not None:
            kwargs["max_len"] = self.max_len
        if self.head_max_len is not None:
            kwargs["head_max_len"] = self.head_max_len
        return kwargs

    def evaluate_state(
        self,
        state: Mapping[str, Any] | str,
        *,
        questions: Mapping[str, Any] | None = None,
    ) -> LayaDecision:
        result = self._load().predict(
            state,
            dict(questions or LAYA_TRADING_QUESTIONS),
            **self._predict_kwargs(),
        )
        return normalize_laya_result(result)

    def evaluate_states(
        self,
        states: Sequence[Mapping[str, Any] | str],
        *,
        questions: Mapping[str, Any] | None = None,
        batch_size: int | None = None,
    ) -> list[LayaDecision]:
        if not states:
            return []
        agent = self._load()
        question_map = dict(questions or LAYA_TRADING_QUESTIONS)
        kwargs = self._predict_kwargs()
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        predict_batch = getattr(agent, "predict_batch", None)
        if callable(predict_batch):
            results = predict_batch(states, question_map, **kwargs)
        else:  # pragma: no cover - compatibility with older/custom runtimes
            results = [agent.predict(state, question_map, **self._predict_kwargs()) for state in states]
        return [normalize_laya_result(result) for result in results]

    # Legacy v0.3 compatibility. New V0.8 code should use evaluate_state(s).
    def evaluate_trade(self, symbol: str, features: Mapping[str, Any], alpha_probability: float) -> dict:
        state = {
            "symbol": symbol,
            "return_1m": float(features.get("return_1m", 0.0)),
            "return_3m": float(features.get("return_3m", 0.0)),
            "return_5m": float(features.get("return_5m", 0.0)),
            "ema_distance": float(features.get("ema_distance", 0.0)),
            "vwap_distance": float(features.get("vwap_distance", 0.0)),
            "relative_volume": float(features.get("relative_volume", 0.0)),
            "range": float(features.get("range", 0.0)),
            "alpha_probability": float(alpha_probability),
        }
        # Preserve the raw shape expected by strategy/decision.py.
        agent = self._load()
        legacy_questions = {
            "action": LAYA_TRADING_QUESTIONS["action"],
            "high_risk": LAYA_TRADING_QUESTIONS["risk_concern"],
        }
        return agent.predict(state, legacy_questions, **self._predict_kwargs())
