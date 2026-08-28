from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a Gate A configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class BudgetConfig:
    hard_limit_usd: float
    reserve_margin: float
    # Pinned CNY-to-USD conversion used when a provider publishes prices in CNY.
    # The ledger itself stays in USD so the Gate A hard cap keeps one unit.
    cny_per_usd: float = 1.0


@dataclass(frozen=True)
class OpenRouterConfig:
    base_url: str
    timeout_seconds: int
    max_retries: int
    app_title: str


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    timeout_seconds: int
    max_retries: int
    api_key_env: str
    app_title: str = ""


@dataclass(frozen=True)
class RoleConfig:
    candidates: tuple[str, ...]
    max_input_tokens: int
    max_output_tokens: int
    required_parameters: tuple[str, ...]


@dataclass(frozen=True)
class RunnerConfig:
    image: str
    timeout_seconds: int
    cpus: float
    memory: str
    pids_limit: int
    docker_command: tuple[str, ...]


@dataclass(frozen=True)
class GateAConfig:
    topic: str
    bfts_nodes: int
    random_seed: int


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    display_name: str
    budget: BudgetConfig
    provider: ProviderConfig
    # Static catalog for providers without a public priced /models endpoint.
    # Entries map model_id -> {context_length, price_cny_per_m_input, price_cny_per_m_output}.
    models: dict[str, dict[str, Any]]
    openrouter: OpenRouterConfig
    roles: dict[str, RoleConfig]
    runner: RunnerConfig
    gate_a: GateAConfig


REQUIRED_ROLES = {"ideation", "experiment_code", "plotting", "paper_writer", "reviewer"}


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing {context}.{key}")
    return mapping[key]


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")

    budget_raw = _require(raw, "budget", "root")
    openrouter_raw = raw.get("openrouter")
    roles_raw = _require(raw, "roles", "root")
    runner_raw = _require(raw, "runner", "root")
    gate_raw = _require(raw, "gate_a", "root")

    provider_raw = raw.get("provider")
    if provider_raw is not None:
        provider_name = str(_require(provider_raw, "name", "provider"))
        if provider_name not in {"openrouter", "zhipu", "openai_compatible"}:
            raise ConfigError(
                "provider.name must be openrouter, zhipu, or openai_compatible, "
                f"got {provider_name!r}"
            )
        provider = ProviderConfig(
            name=provider_name,
            base_url=str(_require(provider_raw, "base_url", "provider")).rstrip("/"),
            timeout_seconds=int(
                _require(provider_raw, "timeout_seconds", "provider")
            ),
            max_retries=int(_require(provider_raw, "max_retries", "provider")),
            api_key_env=str(_require(provider_raw, "api_key_env", "provider")),
            app_title=str(provider_raw.get("app_title", "")),
        )
    else:
        # Legacy Gate A configuration: provider facts live in the openrouter section.
        if openrouter_raw is None:
            raise ConfigError("Missing either provider or openrouter configuration")
        provider = ProviderConfig(
            name="openrouter",
            base_url=str(_require(openrouter_raw, "base_url", "openrouter")).rstrip(
                "/"
            ),
            timeout_seconds=int(
                _require(openrouter_raw, "timeout_seconds", "openrouter")
            ),
            max_retries=int(_require(openrouter_raw, "max_retries", "openrouter")),
            api_key_env="OPENROUTER_API_KEY",
            app_title=str(_require(openrouter_raw, "app_title", "openrouter")),
        )

    if openrouter_raw is None:
        # Providers such as Zhipu do not need an OpenRouter section; keep the legacy
        # field populated so old code paths still load without special cases.
        openrouter_raw = {
            "base_url": provider.base_url,
            "timeout_seconds": provider.timeout_seconds,
            "max_retries": provider.max_retries,
            "app_title": provider.app_title or "path-ai-scientist",
        }

    missing_roles = REQUIRED_ROLES - set(roles_raw)
    if missing_roles:
        raise ConfigError(f"Missing roles: {', '.join(sorted(missing_roles))}")

    roles: dict[str, RoleConfig] = {}
    for role, value in roles_raw.items():
        candidates = tuple(_require(value, "candidates", f"roles.{role}"))
        if not candidates:
            raise ConfigError(f"roles.{role}.candidates cannot be empty")
        if provider.name == "openrouter" and any(
            "/" not in model for model in candidates
        ):
            raise ConfigError(
                f"roles.{role}.candidates must contain OpenRouter model IDs"
            )
        if provider.name != "openrouter" and any(
            not model or model.strip() != model or " " in model for model in candidates
        ):
            raise ConfigError(
                f"roles.{role}.candidates must contain plain model IDs such as glm-5.2"
            )
        roles[role] = RoleConfig(
            candidates=candidates,
            max_input_tokens=int(_require(value, "max_input_tokens", f"roles.{role}")),
            max_output_tokens=int(
                _require(value, "max_output_tokens", f"roles.{role}")
            ),
            required_parameters=tuple(value.get("required_parameters", [])),
        )

    hard_limit = float(_require(budget_raw, "hard_limit_usd", "budget"))
    if not 0 < hard_limit <= 8.0:
        raise ConfigError(
            "Gate A hard_limit_usd must be greater than 0 and at most 8.0"
        )

    cny_per_usd = float(budget_raw.get("cny_per_usd", 1.0))
    if not 0 < cny_per_usd <= 20:
        raise ConfigError("budget.cny_per_usd must be in (0, 20]")

    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        raise ConfigError("models must be a mapping of model_id to catalog entry")
    if provider.name != "openrouter" and not models_raw:
        raise ConfigError(
            "This provider configuration requires a static models catalog with prices"
        )
    models: dict[str, dict[str, Any]] = {}
    for model_id, entry in models_raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"models.{model_id} must be a mapping")
        try:
            context_length = int(_require(entry, "context_length", f"models.{model_id}"))
            input_price = float(
                _require(
                    entry, "price_cny_per_m_input", f"models.{model_id}"
                )
            )
            output_price = float(
                _require(
                    entry, "price_cny_per_m_output", f"models.{model_id}"
                )
            )
        except ConfigError:
            raise
        if context_length <= 0 or input_price < 0 or output_price < 0:
            raise ConfigError(
                f"models.{model_id} has invalid context length or negative prices"
            )
        if input_price == 0 and output_price == 0:
            raise ConfigError(
                f"models.{model_id} has no prices; verify them in the provider console"
            )
        models[str(model_id)] = {
            "context_length": context_length,
            "price_cny_per_m_input": input_price,
            "price_cny_per_m_output": output_price,
            "supported_parameters": list(entry.get("supported_parameters", [])),
        }

    bfts_nodes = int(_require(gate_raw, "bfts_nodes", "gate_a"))
    if bfts_nodes != 1:
        raise ConfigError("Gate A currently permits exactly one BFTS node")

    docker_command = tuple(runner_raw.get("docker_command", ["docker"]))
    if not docker_command:
        raise ConfigError("runner.docker_command cannot be empty")

    return AppConfig(
        schema_version=int(_require(raw, "schema_version", "root")),
        display_name=str(_require(raw, "display_name", "root")),
        budget=BudgetConfig(
            hard_limit_usd=hard_limit,
            reserve_margin=float(_require(budget_raw, "reserve_margin", "budget")),
            cny_per_usd=cny_per_usd,
        ),
        provider=provider,
        models=models,
        openrouter=OpenRouterConfig(
            base_url=str(_require(openrouter_raw, "base_url", "openrouter")).rstrip(
                "/"
            ),
            timeout_seconds=int(
                _require(openrouter_raw, "timeout_seconds", "openrouter")
            ),
            max_retries=int(_require(openrouter_raw, "max_retries", "openrouter")),
            app_title=str(_require(openrouter_raw, "app_title", "openrouter")),
        ),
        roles=roles,
        runner=RunnerConfig(
            image=str(_require(runner_raw, "image", "runner")),
            timeout_seconds=int(_require(runner_raw, "timeout_seconds", "runner")),
            cpus=float(_require(runner_raw, "cpus", "runner")),
            memory=str(_require(runner_raw, "memory", "runner")),
            pids_limit=int(_require(runner_raw, "pids_limit", "runner")),
            docker_command=docker_command,
        ),
        gate_a=GateAConfig(
            topic=str(_require(gate_raw, "topic", "gate_a")),
            bfts_nodes=bfts_nodes,
            random_seed=int(_require(gate_raw, "random_seed", "gate_a")),
        ),
    )
