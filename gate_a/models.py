from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .config import AppConfig, RoleConfig


class ModelSelectionError(RuntimeError):
    """Raised when no configured model can safely serve a role."""


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    context_length: int
    prompt_price: Decimal
    completion_price: Decimal
    request_price: Decimal
    supported_parameters: frozenset[str]

    @classmethod
    def from_api(cls, value: dict[str, Any]) -> "ModelInfo":
        pricing = value.get("pricing") or {}

        def price(name: str) -> Decimal:
            raw = pricing.get(name, "0")
            try:
                parsed = Decimal(str(raw))
            except InvalidOperation as exc:
                raise ModelSelectionError(
                    f"Invalid {name} price for {value.get('id')}"
                ) from exc
            if parsed < 0:
                raise ModelSelectionError(
                    f"Negative {name} price for {value.get('id')}"
                )
            return parsed

        return cls(
            model_id=str(value["id"]),
            context_length=int(value.get("context_length") or 0),
            prompt_price=price("prompt"),
            completion_price=price("completion"),
            request_price=price("request"),
            supported_parameters=frozenset(value.get("supported_parameters") or []),
        )

    def maximum_cost(self, role: RoleConfig, margin: float) -> float:
        base = (
            self.prompt_price * role.max_input_tokens
            + self.completion_price * role.max_output_tokens
            + self.request_price
        )
        return float(base * Decimal(str(margin)))


class ModelRegistry:
    def __init__(self, models: list[ModelInfo]):
        self.models = {model.model_id: model for model in models}

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "ModelRegistry":
        values = payload.get("data")
        if not isinstance(values, list):
            raise ModelSelectionError("OpenRouter model response has no data list")
        models: list[ModelInfo] = []
        for value in values:
            try:
                models.append(ModelInfo.from_api(value))
            except (KeyError, TypeError, ValueError, ModelSelectionError):
                # The public catalog can include beta router entries with sentinel pricing.
                # They are not selectable unless a valid configured candidate parses below.
                continue
        return cls(models)

    @classmethod
    def from_static_catalog(
        cls, catalog: dict[str, dict[str, Any]], cny_per_usd: float
    ) -> "ModelRegistry":
        """Build a registry from config-pinned models for providers without a priced catalog.

        Prices are configured in CNY per million tokens (the unit Zhipu publishes) and
        converted to the per-token USD price the budget ledger uses. The pinned exchange
        rate is an engineering approximation recorded in the run, not a market feed.
        """
        if not catalog:
            raise ModelSelectionError("Static model catalog is empty")
        if not 0 < cny_per_usd <= 20:
            raise ModelSelectionError("cny_per_usd must be in (0, 20]")
        tokens_per_million = Decimal(1_000_000)
        rate = Decimal(str(cny_per_usd))
        models: list[ModelInfo] = []
        for model_id, entry in catalog.items():
            try:
                models.append(
                    ModelInfo(
                        model_id=str(model_id),
                        context_length=int(entry["context_length"]),
                        prompt_price=(
                            Decimal(str(entry["price_cny_per_m_input"]))
                            / tokens_per_million
                            / rate
                        ),
                        completion_price=(
                            Decimal(str(entry["price_cny_per_m_output"]))
                            / tokens_per_million
                            / rate
                        ),
                        request_price=Decimal("0"),
                        supported_parameters=frozenset(
                            entry.get("supported_parameters", [])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
                raise ModelSelectionError(
                    f"Invalid static catalog entry for {model_id}: {type(exc).__name__}"
                ) from exc
        return cls(models)

    def select_all(self, config: AppConfig) -> dict[str, ModelInfo]:
        selected: dict[str, ModelInfo] = {}
        for role_name, role in config.roles.items():
            selected[role_name] = self.select(role_name, role)

        writer = selected["paper_writer"].model_id
        reviewer = selected["reviewer"].model_id
        if writer == reviewer:
            raise ModelSelectionError(
                "paper_writer and reviewer resolved to the same model; independent review is required"
            )
        return selected

    def select(self, role_name: str, role: RoleConfig) -> ModelInfo:
        rejected: list[str] = []
        for candidate in role.candidates:
            model = self.models.get(candidate)
            if model is None:
                rejected.append(f"{candidate}: unavailable")
                continue
            if model.context_length < role.max_input_tokens + role.max_output_tokens:
                rejected.append(f"{candidate}: context too small")
                continue
            missing = set(role.required_parameters) - model.supported_parameters
            if missing:
                rejected.append(f"{candidate}: missing {','.join(sorted(missing))}")
                continue
            if model.prompt_price == 0 and model.completion_price == 0:
                rejected.append(f"{candidate}: missing price metadata")
                continue
            return model
        raise ModelSelectionError(
            f"No model for role {role_name}: {'; '.join(rejected)}"
        )
