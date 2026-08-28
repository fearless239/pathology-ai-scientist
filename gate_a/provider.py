from __future__ import annotations

import json
import hashlib
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .budget import BudgetLedger
from .config import AppConfig
from .models import ModelInfo


class ProviderError(RuntimeError):
    """Raised for malformed or failed provider calls."""


class ProviderOutcomeUnknown(ProviderError):
    """A request may have been billed; never retry or release its reservation."""


def _await_cached_response(
    response_path: Path, attempts: int = 30, interval: float = 2.0
) -> "ProviderResult | None":
    """Wait for a concurrent identical request to publish its cached response."""
    for attempt in range(attempts):
        if response_path.exists():
            cached = json.loads(response_path.read_text(encoding="utf-8"))
            return ProviderResult(cached["value"], cached["metadata"])
        if attempt + 1 < attempts:
            time.sleep(interval)
    return None


class ChatProvider(Protocol):
    selected_models: dict[str, ModelInfo]

    def call_text(
        self, role: str, request_id: str, system: str, prompt: str
    ) -> tuple[str, dict[str, Any]]: ...

    def call_json(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ProviderError(
            "OPENROUTER_API_KEY is not present in the process environment"
        )
    if len(key) < 12:
        raise ProviderError("OPENROUTER_API_KEY is malformed")
    return key


def load_provider_key(config: AppConfig) -> str:
    """Load the configured provider key from the process environment only."""
    env_name = config.provider.api_key_env
    key = os.environ.get(env_name, "")
    if not key:
        raise ProviderError(f"{env_name} is not present in the process environment")
    if len(key) < 12:
        raise ProviderError(f"{env_name} is malformed")
    return key


def _safe_request_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ProviderError(f"Unsafe request ID: {value!r}")
    if len(value) > 100:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        value = f"{value[:75]}-{digest}"
    return value


def _empty_text_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "IMPORTANT RETRY: The previous provider response contained no text. "
        "Return the complete requested document as plain text now. Do not return an empty "
        "message, a tool call, an acknowledgement, or only a title."
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _usage_dict(raw: dict[str, Any]) -> dict[str, Any]:
    usage = raw.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cost": usage.get("cost"),
        "cost_details": usage.get("cost_details") or {},
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from model text, tolerating prose wrappers."""
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, dict):
                        return candidate
                    break
    return None


@dataclass(frozen=True)
class ProviderResult:
    value: str | dict[str, Any]
    metadata: dict[str, Any]


class OpenRouterProvider:
    """OpenRouter adapter with pre-request reservation and response caching."""

    def __init__(
        self,
        config: AppConfig,
        selected_models: dict[str, ModelInfo],
        ledger: BudgetLedger,
        response_dir: Path,
    ):
        self.config = config
        self.selected_models = selected_models
        self.ledger = ledger
        self.response_dir = response_dir
        self._key = load_openrouter_key()

    def call_text(
        self, role: str, request_id: str, system: str, prompt: str
    ) -> tuple[str, dict[str, Any]]:
        for attempt in range(3):
            active_id = request_id if attempt == 0 else f"{request_id}-empty-retry-{attempt}"
            active_prompt = prompt if attempt == 0 else _empty_text_retry_prompt(prompt)
            result = self._call(role, active_id, system, active_prompt, None, None)
            if isinstance(result.value, str) and result.value.strip():
                metadata = {**result.metadata, "empty_retry_attempt": attempt}
                return result.value, metadata
        raise ProviderError(f"Request {request_id} returned empty text after 3 attempts")

    def call_json(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: ProviderError | None = None
        for attempt in range(3):
            active_id = request_id if attempt == 0 else f"{request_id}-json-v2-retry-{attempt}"
            active_prompt = prompt
            if attempt:
                active_prompt += (
                    "\n\nPrevious structured output was empty or invalid. Return the requested "
                    "function arguments as one complete JSON object only."
                )
            try:
                result = self._call(role, active_id, system, active_prompt, function_name, schema)
            except ProviderOutcomeUnknown:
                raise
            except ProviderError as error:
                last_error = error
                continue
            if isinstance(result.value, dict):
                return result.value, {**result.metadata, "json_retry_attempt": attempt}
            last_error = ProviderError(f"Request {active_id} did not return a JSON object")
        raise last_error or ProviderError(f"Request {request_id} did not return a JSON object")

    def _call(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str | None,
        schema: dict[str, Any] | None,
    ) -> ProviderResult:
        request_id = _safe_request_id(request_id)
        response_path = self.response_dir / f"{request_id}.json"
        if response_path.exists():
            cached = json.loads(response_path.read_text(encoding="utf-8"))
            self.ledger.recover_cached(request_id, cached["metadata"])
            return ProviderResult(cached["value"], cached["metadata"])

        role_config = self.config.roles[role]
        model = self.selected_models[role]
        reservation = model.maximum_cost(role_config, self.config.budget.reserve_margin)
        was_reserved = self.ledger.reserve(
            request_id,
            reservation,
            {
                "role": role,
                "model": model.model_id,
                "max_output_tokens": role_config.max_output_tokens,
            },
        )
        if not was_reserved:
            recovered = _await_cached_response(response_path)
            if recovered is not None:
                self.ledger.recover_cached(request_id, recovered.metadata)
                return recovered
            raise ProviderOutcomeUnknown(
                f"Request {request_id} is reserved or settled but its cached response is missing; reconciliation required"
            )

        body: dict[str, Any] = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": role_config.max_output_tokens,
            "temperature": 0.2,
            "provider": {"allow_fallbacks": False},
        }
        if function_name and schema:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "Return the required structured result.",
                        "parameters": schema,
                    },
                }
            ]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": function_name},
            }

        try:
            raw = self._post_json("/chat/completions", body)
            if raw.get("error"):
                usage = _usage_dict(raw)
                actual_cost = self._actual_cost(model, usage)
                self.ledger.settle(request_id, actual_cost, usage)
                raise ProviderError(
                    f"OpenRouter returned an in-band generation error for {request_id}"
                )
            choice = raw["choices"][0]["message"]
            if function_name:
                calls = choice.get("tool_calls") or []
                if not calls:
                    raise ProviderError(
                        f"Request {request_id} returned no required tool call"
                    )
                arguments = calls[0]["function"]["arguments"]
                value: str | dict[str, Any] = json.loads(arguments)
            else:
                value = choice.get("content") or ""

            usage = _usage_dict(raw)
            actual_cost = self._actual_cost(model, usage)
            metadata = {
                "request_id": raw.get("id"),
                "requested_model": model.model_id,
                "resolved_model": raw.get("model"),
                "usage": usage,
                "actual_cost_usd": actual_cost,
                "cached": False,
            }
            _atomic_write_json(response_path, {"value": value, "metadata": metadata})
            self.ledger.settle(request_id, actual_cost, usage)
            return ProviderResult(value, metadata)
        except Exception as exc:
            if 'actual_cost' in locals() and 'usage' in locals():
                # The provider answered: never turn a persistence error into a paid retry.
                self.ledger.recover_cached(request_id, {'actual_cost_usd': actual_cost, 'usage': usage})
            elif not isinstance(exc, ProviderOutcomeUnknown):
                self.ledger.release(request_id, type(exc).__name__)
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                f"OpenRouter request {request_id} failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _actual_cost(model: ModelInfo, usage: dict[str, Any]) -> float:
        if usage["cost"] is not None:
            return float(usage["cost"])
        return float(
            model.prompt_price * usage["prompt_tokens"]
            + model.completion_price * usage["completion_tokens"]
            + model.request_price
        )

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.config.openrouter.base_url + path
        data = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.openrouter.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": self.config.openrouter.app_title,
                    # Identical explicit retries can reuse OpenRouter's response cache. Network
                    # timeouts are not retried because the API documents no chat idempotency key.
                    "X-OpenRouter-Cache": "true",
                    "X-OpenRouter-Cache-TTL": "86400",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.openrouter.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 408 or exc.code >= 500:
                    raise ProviderOutcomeUnknown(f"HTTP {exc.code}: billing outcome unknown; reconciliation required") from exc
                if exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise ProviderOutcomeUnknown("Transport/response failure: billing outcome unknown; reconciliation required") from exc
            if attempt < self.config.openrouter.max_retries:
                retry_after = getattr(last_error, "headers", {}).get("Retry-After")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = min(2**attempt, 8)
                time.sleep(min(max(delay, 0), 60))
        status = getattr(last_error, "code", "network")
        detail = f": {str(last_error)[:300]}" if status == "network" and last_error else ""
        raise ProviderError(f"OpenRouter HTTP request failed with status {status}{detail}")


class ZhipuProvider:
    """OpenAI-compatible adapter for Zhipu (open.bigmodel.cn) with pre-request reservation.

    Zhipu publishes no priced machine-readable model catalog, so prices come from the
    pinned static catalog in the configuration and costs are computed locally from token
    usage. The request body intentionally omits OpenRouter-specific fields and headers.
    """

    def __init__(
        self,
        config: AppConfig,
        selected_models: dict[str, ModelInfo],
        ledger: BudgetLedger,
        response_dir: Path,
    ):
        self.config = config
        self.selected_models = selected_models
        self.ledger = ledger
        self.response_dir = response_dir
        self._key = load_provider_key(config)

    def call_text(
        self, role: str, request_id: str, system: str, prompt: str
    ) -> tuple[str, dict[str, Any]]:
        for attempt in range(3):
            active_id = request_id if attempt == 0 else f"{request_id}-text-v2-retry-{attempt}"
            active_prompt = prompt if attempt == 0 else _empty_text_retry_prompt(prompt)
            result = self._call(role, active_id, system, active_prompt, None, None)
            if isinstance(result.value, str) and result.value.strip():
                metadata = {**result.metadata, "empty_retry_attempt": attempt}
                return result.value, metadata
        raise ProviderError(f"Request {request_id} returned empty text after 3 attempts")

    def call_json(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: ProviderError | None = None
        for attempt in range(3):
            active_id = request_id if attempt == 0 else f"{request_id}-json-v2-retry-{attempt}"
            active_prompt = prompt
            if attempt:
                active_prompt += (
                    "\n\nPrevious structured output was empty or invalid. Return the requested "
                    "function arguments as one complete JSON object only."
                )
            try:
                result = self._call(role, active_id, system, active_prompt, function_name, schema)
            except ProviderOutcomeUnknown:
                raise
            except ProviderError as error:
                last_error = error
                continue
            if isinstance(result.value, dict):
                return result.value, {**result.metadata, "json_retry_attempt": attempt}
            last_error = ProviderError(f"Request {active_id} did not return a JSON object")
        raise last_error or ProviderError(f"Request {request_id} did not return a JSON object")

    def _call(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str | None,
        schema: dict[str, Any] | None,
    ) -> ProviderResult:
        request_id = _safe_request_id(request_id)
        response_path = self.response_dir / f"{request_id}.json"
        if response_path.exists():
            cached = json.loads(response_path.read_text(encoding="utf-8"))
            self.ledger.recover_cached(request_id, cached["metadata"])
            return ProviderResult(cached["value"], cached["metadata"])

        role_config = self.config.roles[role]
        model = self.selected_models[role]
        reservation = model.maximum_cost(role_config, self.config.budget.reserve_margin)
        was_reserved = self.ledger.reserve(
            request_id,
            reservation,
            {
                "role": role,
                "model": model.model_id,
                "max_output_tokens": role_config.max_output_tokens,
            },
        )
        if not was_reserved:
            recovered = _await_cached_response(response_path)
            if recovered is not None:
                self.ledger.recover_cached(request_id, recovered.metadata)
                return recovered
            raise ProviderOutcomeUnknown(
                f"Request {request_id} is reserved or settled but its cached response is missing; reconciliation required"
            )

        body: dict[str, Any] = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": role_config.max_output_tokens,
            "temperature": 0.2,
        }
        # GLM-5.x defaults to deep thinking and may spend the whole completion
        # budget without emitting content or a tool call. These workflow calls
        # need direct, machine-readable output.
        body["thinking"] = {"type": "disabled"}
        structured_method = "plain_text"
        if function_name and schema:
            # Zhipu documents tools with tool_choice="auto"; a named forced tool_choice is
            # not part of the documented surface, so parse tool calls with a content-JSON
            # fallback instead of forcing the call.
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": "Return the required structured result.",
                        "parameters": schema,
                    },
                }
            ]
            body["tool_choice"] = "auto"
            structured_method = "tool_call_or_content_json"

        try:
            try:
                raw = self._post_json("/chat/completions", body)
            except ProviderOutcomeUnknown:
                raise
            except ProviderError as exc:
                # Subscription coding endpoints may reject tool parameters even though
                # chat completions work. Retry once without tools; the content-JSON
                # fallback below still yields the structured result.
                if function_name and "tools" in body and "status 400" in str(exc):
                    body.pop("tools", None)
                    body.pop("tool_choice", None)
                    raw = self._post_json("/chat/completions", body)
                else:
                    raise
            if raw.get("error"):
                usage = _usage_dict(raw)
                actual_cost = self._actual_cost(model, usage)
                self.ledger.settle(request_id, actual_cost, usage)
                raise ProviderError(
                    f"Zhipu returned an in-band generation error for {request_id}"
                )
            choice = raw["choices"][0]["message"]
            value: str | dict[str, Any]
            if function_name:
                calls = choice.get("tool_calls") or []
                if calls:
                    arguments = calls[0]["function"]["arguments"]
                    try:
                        value = json.loads(arguments)
                    except json.JSONDecodeError:
                        value = _extract_json_object(arguments)
                    structured_method = "tool_call"
                else:
                    content = choice.get("content") or ""
                    value = _extract_json_object(content)
                    if value is None:
                        # Some answers arrive as prose with no JSON at all. Persist the
                        # raw reply for diagnostics and re-ask once with an explicit
                        # JSON-only instruction; both attempts share one reservation and
                        # only the final usage is settled.
                        failure_path = self.response_dir / f"{request_id}.failed.txt"
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        failure_path.write_text(content, encoding="utf-8")
                        repair_body = {
                            key: item
                            for key, item in body.items()
                            if key not in {"tools", "tool_choice"}
                        }
                        repair_body["thinking"] = {"type": "disabled"}
                        repair_body["messages"] = [
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": (
                                    f"{prompt}\n\nIMPORTANT: Respond with ONLY the JSON "
                                    "object for the requested function. No prose and no "
                                    "markdown code fences."
                                ),
                            },
                        ]
                        raw = self._post_json("/chat/completions", repair_body)
                        if raw.get("error"):
                            raise ProviderError(
                                f"Zhipu returned an in-band generation error for {request_id}"
                            )
                        choice = raw["choices"][0]["message"]
                        content = choice.get("content") or ""
                        value = _extract_json_object(content)
                        structured_method = "content_json_repair"
                        if value is None:
                            failure_path.write_text(content, encoding="utf-8")
                            # The provider completed both paid attempts even though
                            # neither produced usable JSON. Account for that usage
                            # before surfacing the parse failure; the outer release
                            # becomes a no-op once the reservation is settled.
                            failed_usage = _usage_dict(raw)
                            failed_cost = self._actual_cost(model, failed_usage)
                            self.ledger.settle(request_id, failed_cost, failed_usage)
                            raise ProviderError(
                                f"Request {request_id} returned no parseable JSON object; "
                                f"the raw reply was saved next to the response cache as "
                                f"{failure_path.name}"
                            )
                    else:
                        structured_method = "content_json"
            else:
                value = choice.get("content") or ""

            usage = _usage_dict(raw)
            actual_cost = self._actual_cost(model, usage)
            metadata = {
                "request_id": raw.get("id"),
                "requested_model": model.model_id,
                "resolved_model": raw.get("model") or model.model_id,
                "usage": usage,
                "actual_cost_usd": actual_cost,
                "structured_method": structured_method,
                "cached": False,
            }
            _atomic_write_json(response_path, {"value": value, "metadata": metadata})
            self.ledger.settle(request_id, actual_cost, usage)
            return ProviderResult(value, metadata)
        except Exception as exc:
            if 'actual_cost' in locals() and 'usage' in locals():
                # The provider answered: never turn a persistence error into a paid retry.
                self.ledger.recover_cached(request_id, {'actual_cost_usd': actual_cost, 'usage': usage})
            elif not isinstance(exc, ProviderOutcomeUnknown):
                self.ledger.release(request_id, type(exc).__name__)
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                f"Zhipu request {request_id} failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _actual_cost(model: ModelInfo, usage: dict[str, Any]) -> float:
        # Zhipu does not return a server-computed cost; use the pinned static prices.
        return float(
            model.prompt_price * usage["prompt_tokens"]
            + model.completion_price * usage["completion_tokens"]
            + model.request_price
        )

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.config.provider.base_url + path
        data = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.provider.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.provider.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 408 or exc.code >= 500:
                    raise ProviderOutcomeUnknown(f"HTTP {exc.code}: billing outcome unknown; reconciliation required") from exc
                if exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise ProviderOutcomeUnknown("Transport/response failure: billing outcome unknown; reconciliation required") from exc
            if attempt < self.config.provider.max_retries:
                retry_after = getattr(last_error, "headers", {}).get("Retry-After")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = min(2**attempt, 8)
                time.sleep(min(max(delay, 0), 60))
        status = getattr(last_error, "code", "network")
        detail = ""
        if isinstance(last_error, urllib.error.HTTPError):
            try:
                payload = json.loads(last_error.read().decode("utf-8"))
                message = payload.get("error", {}).get("message")
                if message:
                    detail = f": {str(message)[:300]}"
            except Exception:
                detail = ""
        elif last_error is not None:
            detail = f": {str(last_error)[:300]}"
        raise ProviderError(f"Zhipu HTTP request failed with status {status}{detail}")


class FixtureProvider:
    """Deterministic offline responses used to test orchestration without API charges."""

    def __init__(self, selected_models: dict[str, ModelInfo]):
        self.selected_models = selected_models

    def call_text(
        self, role: str, request_id: str, system: str, prompt: str
    ) -> tuple[str, dict[str, Any]]:
        del system, prompt
        if role == "experiment_code":
            value = """A deterministic synthetic classification task provides a fast engineering baseline. The code fixes all random seeds and creates two informative features. It uses a closed-form linear score to avoid external data. Validation accuracy is measured over a held-out split. Loss and accuracy histories are stored for plotting. Raw predictions and labels are retained in the experiment data. The run is intentionally small and makes no scientific novelty claim. All artifacts are written under the required working directory.
```python
import json
import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
rng = np.random.default_rng(7)
x = rng.normal(size=(240, 2))
y = (x[:, 0] + 0.55 * x[:, 1] > 0).astype(int)
score = x[:, 0] + 0.45 * x[:, 1]
pred = (score > 0).astype(int)
accuracy = float(np.mean(pred == y))
losses = [0.62, 0.44, 0.31, 0.25]
accuracies = [0.72, 0.83, 0.90, accuracy]
experiment_data = {
    "synthetic_binary": {
        "metrics": {"validation_accuracy": accuracies},
        "losses": {"validation": losses},
        "predictions": pred,
        "ground_truth": y,
        "epochs": [1, 2, 3, 4],
    }
}
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
with open(os.path.join(working_dir, "metrics.json"), "w", encoding="utf-8") as handle:
    json.dump({"primary_metric": "validation_accuracy", "value": accuracy, "n": len(y)}, handle)
print(f"validation_accuracy={accuracy:.6f}")
```
"""
        elif role == "plotting":
            value = """The plot reads only the stored experiment data. It displays validation accuracy by epoch with explicit axes and title. No values are invented or extrapolated. The PNG is saved in the required working directory.
```python
import os
import matplotlib.pyplot as plt
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
data = np.load(os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True).item()
run = data["synthetic_binary"]
plt.figure(figsize=(5, 3))
plt.plot(run["epochs"], run["metrics"]["validation_accuracy"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Gate A synthetic baseline")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "validation_accuracy.png"), dpi=120)
plt.close()
print("plot=validation_accuracy.png")
```
"""
        else:
            raise ProviderError(f"No offline text fixture for role {role}")
        return value, self._meta(role, request_id)

    def call_json(
        self,
        role: str,
        request_id: str,
        system: str,
        prompt: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del system, prompt, function_name, schema
        if role == "ideation":
            value = {
                "Name": "deterministic_feature_transform_smoke",
                "Title": "A Minimal Deterministic Feature-Transformation Smoke Study",
                "Short Hypothesis": "A fixed linear feature transformation changes held-out accuracy.",
                "Related Work": "No literature claim is made in this engineering smoke test.",
                "Abstract": "We exercise a reduced automated research chain on synthetic data.",
                "Experiments": "Run one deterministic synthetic binary classification baseline.",
                "Risk Factors and Limitations": "Synthetic data, one run, no external validity.",
            }
        elif role == "paper_writer":
            value = {
                "title": "A Minimal Deterministic Engineering Smoke Study",
                "abstract": "We report a synthetic experiment produced by a constrained AI-Scientist-v2 engineering reproduction. The result is preliminary and requires human review.",
                "introduction": "The purpose of this study is to verify an end-to-end automated engineering chain, not to establish scientific novelty.",
                "method": "A deterministic two-feature binary dataset was generated locally. Agent-generated Python ran in an offline container and emitted structured metrics.",
                "results": "The pipeline produced a non-empty accuracy metric and a plot derived from saved raw experiment data.",
                "limitations": "This is a single synthetic smoke run. It has no clinical or real-world validity and must not be interpreted as a substantive research result.",
                "conclusion": "The reduced chain completed its engineering objectives. Human review is required before any reuse of the manuscript.",
            }
        elif role == "reviewer":
            value = {
                "overall_score": 4,
                "decision": "engineering_smoke_pass",
                "summary": "The manuscript is structurally complete for a smoke test but contains no scientific novelty claim.",
                "strengths": [
                    "Clear scope",
                    "Explicit limitations",
                    "Traceable artifact claim",
                ],
                "weaknesses": ["Synthetic data", "Single run", "No external validity"],
                "required_changes": [
                    "Retain the human-review and AI-generation disclosures"
                ],
            }
        else:
            raise ProviderError(f"No offline JSON fixture for role {role}")
        return value, self._meta(role, request_id)

    def _meta(self, role: str, request_id: str) -> dict[str, Any]:
        model = self.selected_models[role].model_id
        return {
            "request_id": f"fixture-{request_id}",
            "requested_model": model,
            "resolved_model": model,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
            },
            "actual_cost_usd": 0.0,
            "cached": False,
            "fixture": True,
        }


def fetch_model_catalog(config: AppConfig, key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        config.openrouter.base_url + "/models",
        method="GET",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.openrouter.timeout_seconds
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ProviderError(
            f"Unable to fetch OpenRouter model catalog: {type(exc).__name__}"
        ) from exc


def probe_zhipu_model_list(config: AppConfig) -> list[str] | None:
    """Best-effort free probe of the OpenAI-compatible model list endpoint.

    Returns model IDs when the endpoint is available, or None when the provider does not
    expose a usable list (in which case availability is only proven by the paid smoke).
    Authentication failures raise because a preflight with a bad key must not pass.
    """
    key = load_provider_key(config)
    request = urllib.request.Request(
        config.provider.base_url + "/models",
        method="GET",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.provider.timeout_seconds
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ProviderError(
                f"Zhipu rejected the configured API key with status {exc.code}"
            ) from exc
        return None
    except Exception:
        return None
    values = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return None
    return [
        str(item.get("id"))
        for item in values
        if isinstance(item, dict) and item.get("id")
    ]
