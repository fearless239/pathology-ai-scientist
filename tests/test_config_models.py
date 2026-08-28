from decimal import Decimal

import pytest

from gate_a.config import ConfigError, load_config
from gate_a.models import ModelInfo, ModelRegistry, ModelSelectionError


def test_checked_in_config_is_gate_a_bounded(project_root):
    config = load_config(project_root / "configs" / "gate_a.yaml")
    assert config.budget.hard_limit_usd == 8.0
    assert config.gate_a.bfts_nodes == 1
    assert config.runner.docker_command == ("docker",)


def test_model_registry_checks_capabilities_and_review_independence(project_root):
    config = load_config(project_root / "configs" / "gate_a.yaml")
    ids = {candidate for role in config.roles.values() for candidate in role.candidates}
    models = [
        ModelInfo(
            model_id=model_id,
            context_length=200_000,
            prompt_price=Decimal("0.0000001"),
            completion_price=Decimal("0.0000002"),
            request_price=Decimal("0"),
            supported_parameters=frozenset({"tools", "tool_choice"}),
        )
        for model_id in ids
    ]
    selected = ModelRegistry(models).select_all(config)
    assert selected["paper_writer"].model_id != selected["reviewer"].model_id


def test_missing_tool_support_is_rejected(project_root):
    config = load_config(project_root / "configs" / "gate_a.yaml")
    role = config.roles["ideation"]
    model = ModelInfo(
        model_id=role.candidates[0],
        context_length=200_000,
        prompt_price=Decimal("0.1"),
        completion_price=Decimal("0.1"),
        request_price=Decimal("0"),
        supported_parameters=frozenset(),
    )
    with pytest.raises(ModelSelectionError):
        ModelRegistry([model]).select("ideation", role)


def test_invalid_unrelated_catalog_entry_does_not_block_candidates():
    registry = ModelRegistry.from_api(
        {
            "data": [
                {
                    "id": "openrouter/auto-beta",
                    "context_length": 1000,
                    "pricing": {"prompt": "-1", "completion": "0"},
                },
                {
                    "id": "vendor/valid",
                    "context_length": 1000,
                    "pricing": {"prompt": "0.1", "completion": "0.2"},
                },
            ]
        }
    )
    assert "vendor/valid" in registry.models
    assert "openrouter/auto-beta" not in registry.models


def test_config_rejects_more_than_eight_dollars(tmp_path, project_root):
    text = (project_root / "configs" / "gate_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        text.replace("hard_limit_usd: 8.0", "hard_limit_usd: 8.01"), encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_config(path)
