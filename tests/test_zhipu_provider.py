from decimal import Decimal

import pytest

from gate_a.budget import BudgetLedger
from gate_a.config import load_config
from gate_a.models import ModelRegistry, ModelSelectionError
from gate_a.provider import ProviderError, ZhipuProvider, _extract_json_object


def _load(project_root):
    return load_config(project_root / "configs" / "gate_a_llm.yaml")


def _selected(config):
    registry = ModelRegistry.from_static_catalog(
        config.models, config.budget.cny_per_usd
    )
    return registry.select_all(config)


def test_unknown_transport_outcome_is_not_retried_or_released(project_root, tmp_path, monkeypatch):
    from gate_a.provider import ProviderOutcomeUnknown
    config = _load(project_root)
    monkeypatch.setenv('PARATERA_API_KEY', 'paratera-test-key-123456')
    ledger = BudgetLedger(tmp_path / 'budget.json', config.budget.hard_limit_usd)
    provider = ZhipuProvider(config, _selected(config), ledger, tmp_path / 'responses')
    calls = []
    def timeout(*args, **kwargs):
        calls.append(1)
        raise TimeoutError('response lost')
    monkeypatch.setattr('urllib.request.urlopen', timeout)
    with pytest.raises(ProviderOutcomeUnknown, match='outcome unknown'):
        provider.call_json('paper_writer', 'uncertain', 'system', 'prompt', 'result', {'type': 'object'})
    assert calls == [1]
    assert ledger.snapshot().reserved_usd > 0
    monkeypatch.setattr('gate_a.provider._await_cached_response', lambda *args: None)
    with pytest.raises(ProviderOutcomeUnknown, match='reconciliation'):
        provider.call_json('paper_writer', 'uncertain', 'system', 'prompt', 'result', {'type': 'object'})
    assert calls == [1]


def _provider(project_root, tmp_path, monkeypatch, responses):
    config = _load(project_root)
    monkeypatch.setenv("PARATERA_API_KEY", "paratera-test-key-123456")
    ledger = BudgetLedger(tmp_path / "budget.json", config.budget.hard_limit_usd)
    provider = ZhipuProvider(config, _selected(config), ledger, tmp_path / "responses")

    calls = []

    def fake_post(path, body):
        calls.append({"path": path, "body": body})
        return responses.pop(0)

    monkeypatch.setattr(
        ZhipuProvider, "_post_json", lambda self, path, body: fake_post(path, body)
    )
    return provider, ledger, calls


def test_call_json_repairs_prose_response(project_root, tmp_path, monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    responses = [
        {
            "id": "resp-prose",
            "model": "GLM-5.2",
            "choices": [{"message": {"content": "抱歉，这个要求我需要更多背景信息才能回答。"}}],
            "usage": usage,
        },
        {
            "id": "resp-repair",
            "model": "GLM-5.2",
            "choices": [
                {"message": {"content": '```json\n{"name": "directed_study"}\n```'}}
            ],
            "usage": usage,
        },
    ]
    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, responses)
    value, metadata = provider.call_json(
        "ideation",
        "req-repair",
        "system",
        "prompt",
        "FinalizeIdea",
        {"type": "object", "properties": {"name": {"type": "string"}}},
    )
    assert value == {"name": "directed_study"}
    assert metadata["structured_method"] == "content_json_repair"
    assert len(calls) == 2
    assert "IMPORTANT" in calls[1]["body"]["messages"][1]["content"]
    assert "tools" not in calls[1]["body"]
    assert (tmp_path / "responses" / "req-repair.failed.txt").is_file()
    assert ledger.snapshot().spent_usd > 0


def test_zhipu_config_loads_with_static_catalog(project_root):
    config = _load(project_root)
    assert config.provider.name == "openai_compatible"
    assert config.provider.api_key_env == "PARATERA_API_KEY"
    assert config.provider.base_url == "https://llmapi.paratera.com/v1"
    assert set(config.models) == {"GLM-5.2", "GLM-5.1"}
    selected = _selected(config)
    assert selected["paper_writer"].model_id != selected["reviewer"].model_id


def test_static_catalog_converts_cny_prices_to_usd_per_token(project_root):
    config = _load(project_root)
    registry = ModelRegistry.from_static_catalog(
        config.models, config.budget.cny_per_usd
    )
    model = registry.models["GLM-5.2"]
    expected_input = Decimal("8.0") / Decimal(1_000_000) / Decimal("7.1")
    expected_output = Decimal("28.0") / Decimal(1_000_000) / Decimal("7.1")
    assert model.prompt_price == expected_input
    assert model.completion_price == expected_output
    assert model.context_length == 1_048_576


def test_static_catalog_rejects_model_with_insufficient_context(project_root):
    config = _load(project_root)
    glm51 = ModelRegistry.from_static_catalog(
        config.models, config.budget.cny_per_usd
    ).models["GLM-5.1"]
    tiny = type(glm51)(
        model_id="GLM-5.1",
        context_length=1000,
        prompt_price=glm51.prompt_price,
        completion_price=glm51.completion_price,
        request_price=glm51.request_price,
        supported_parameters=glm51.supported_parameters,
    )
    with pytest.raises(ModelSelectionError):
        ModelRegistry([tiny]).select_all(config)


def test_extract_json_object_handles_prose_strings_and_nested_braces():
    assert _extract_json_object('prefix {"a": {"b": "}"}, "c": 1} suffix') == {
        "a": {"b": "}"},
        "c": 1,
    }
    assert _extract_json_object("no json here") is None
    assert _extract_json_object('broken {"a": } then {"ok": 2}') == {"ok": 2}


def test_call_json_tool_call_settles_and_caches(project_root, tmp_path, monkeypatch):
    responses = [
        {
            "id": "zhipu-1",
            "model": "GLM-5.2",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"Name": "x", "Title": "y"}'
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
    ]
    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, responses)
    schema = {
        "type": "object",
        "properties": {"Name": {"type": "string"}, "Title": {"type": "string"}},
        "required": ["Name", "Title"],
    }
    value, metadata = provider.call_json(
        "ideation", "ideation-000", "system", "prompt", "submit_idea", schema
    )
    assert value == {"Name": "x", "Title": "y"}
    assert metadata["structured_method"] == "tool_call"
    assert len(calls) == 1
    body = calls[0]["body"]
    assert body["tool_choice"] == "auto"
    assert "provider" not in body
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 3500

    per_million = Decimal(1_000_000) * Decimal("7.1")
    expected = float(
        Decimal("8.0") * 100 / per_million + Decimal("28.0") * 50 / per_million
    )
    assert metadata["actual_cost_usd"] == pytest.approx(expected)
    # The ledger settles with 10 decimal places, so compare with the rounded value.
    assert ledger.snapshot().spent_usd == round(expected, 10)

    value2, metadata2 = provider.call_json(
        "ideation", "ideation-000", "system", "prompt", "submit_idea", schema
    )
    assert value2 == value
    assert metadata2 == metadata
    assert len(calls) == 1


def test_call_json_content_fallback(project_root, tmp_path, monkeypatch):
    responses = [
        {
            "id": "zhipu-2",
            "model": "GLM-5.2",
            "choices": [
                {
                    "message": {
                        "content": 'Sure! {"Name": "a", "Title": "b"} hope that helps'
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ]
    provider, _, _ = _provider(project_root, tmp_path, monkeypatch, responses)
    value, metadata = provider.call_json(
        "ideation", "ideation-001", "system", "prompt", "submit_idea", {"type": "object"}
    )
    assert value == {"Name": "a", "Title": "b"}
    assert metadata["structured_method"] == "content_json"


def test_call_json_retries_without_tools_on_400(project_root, tmp_path, monkeypatch):
    config = _load(project_root)
    monkeypatch.setenv("PARATERA_API_KEY", "paratera-test-key-123456")
    ledger = BudgetLedger(tmp_path / "budget.json", config.budget.hard_limit_usd)
    provider = ZhipuProvider(config, _selected(config), ledger, tmp_path / "responses")

    calls = []

    def fake_post(path, body):
        calls.append(dict(body))
        if len(calls) == 1:
            raise ProviderError(
                "Zhipu HTTP request failed with status 400: tools is not supported"
            )
        return {
            "id": "zhipu-4",
            "model": "GLM-5.2",
            "choices": [{"message": {"content": '{"Name": "n", "Title": "t"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(
        ZhipuProvider, "_post_json", lambda self, path, body: fake_post(path, body)
    )
    value, metadata = provider.call_json(
        "ideation", "ideation-002", "system", "prompt", "submit_idea", {"type": "object"}
    )
    assert value == {"Name": "n", "Title": "t"}
    assert metadata["structured_method"] == "content_json"
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert "tool_choice" not in calls[1]


def test_empty_text_releases_reservation(project_root, tmp_path, monkeypatch):
    responses = [
        {
            "id": "zhipu-3",
            "model": "GLM-5.2",
            "choices": [{"message": {"content": "   "}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
    ]
    provider, ledger, _ = _provider(project_root, tmp_path, monkeypatch, responses)
    with pytest.raises(ProviderError):
        provider.call_text("experiment_code", "code-000", "system", "prompt")
    snapshot = ledger.snapshot()
    # Empty output still consumed input tokens, so the request is settled (paid),
    # not released. No reservation may remain open.
    assert snapshot.spent_usd > 0.0
    assert snapshot.reserved_usd == 0.0


def test_empty_text_uses_stable_retry_and_cache(project_root, tmp_path, monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    responses = [
        {
            "id": "empty-first",
            "model": "GLM-5.2",
            "choices": [{"message": {"content": ""}}],
            "usage": usage,
        },
        {
            "id": "retry-success",
            "model": "GLM-5.2",
            "choices": [{"message": {"content": "# Complete paper\n\nBody"}}],
            "usage": usage,
        },
    ]
    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, responses)

    value, metadata = provider.call_text(
        "paper_writer", "paper-empty", "system", "Write the paper"
    )

    assert value.startswith("# Complete paper")
    assert metadata["empty_retry_attempt"] == 1
    assert len(calls) == 2
    assert "IMPORTANT RETRY" in calls[1]["body"]["messages"][1]["content"]
    assert calls[0]["body"]["thinking"] == {"type": "disabled"}
    assert calls[1]["body"]["thinking"] == {"type": "disabled"}
    assert (tmp_path / "responses" / "paper-empty.json").is_file()
    assert (tmp_path / "responses" / "paper-empty-text-v2-retry-1.json").is_file()
    assert ledger.snapshot().spent_usd > 0
    cached_value, cached_metadata = provider.call_text(
        "paper_writer", "paper-empty", "system", "Write the paper"
    )
    assert cached_value == value
    assert cached_metadata["empty_retry_attempt"] == 1
    assert len(calls) == 2

