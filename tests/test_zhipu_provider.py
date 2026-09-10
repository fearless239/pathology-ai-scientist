from decimal import Decimal
import json

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


def test_unknown_transport_outcome_recovers_once_on_a_later_call(project_root, tmp_path, monkeypatch):
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
    diagnostic_path = tmp_path / 'responses' / 'diagnostics' / 'uncertain.json'
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    assert diagnostic['state'] == 'outcome_unknown'
    assert diagnostic['attempts'][0]['failure_category'] == 'read_timeout'
    assert diagnostic['input_tokens'] is None
    assert 'prompt' not in diagnostic_path.read_text(encoding='utf-8')
    first_reservation = ledger.snapshot().reserved_usd
    assert first_reservation > 0
    assert ledger.request_record('uncertain')['state'] == 'outcome_unknown'
    # A pre-recovery-version ledger kept the same diagnosed request as reserved.
    # The durable diagnostic, rather than age or guesswork, authorizes migration.
    ledger_raw = json.loads(ledger.path.read_text(encoding='utf-8'))
    ledger_raw['requests']['uncertain']['state'] = 'reserved'
    ledger.path.write_text(json.dumps(ledger_raw), encoding='utf-8')
    with pytest.raises(ProviderOutcomeUnknown, match='outcome unknown'):
        provider.call_json('paper_writer', 'uncertain', 'system', 'prompt', 'result', {'type': 'object'})
    assert calls == [1, 1]
    assert ledger.request_record('uncertain-outcome-recovery-1')['state'] == 'outcome_unknown'
    assert ledger.snapshot().reserved_usd == pytest.approx(first_reservation * 2)
    with pytest.raises(ProviderOutcomeUnknown, match='exhausted 1'):
        provider.call_json('paper_writer', 'uncertain', 'system', 'prompt', 'result', {'type': 'object'})
    assert calls == [1, 1]


def test_success_diagnostic_records_transport_and_usage_without_prompt(
    project_root, tmp_path, monkeypatch
):
    config = _load(project_root)
    monkeypatch.setenv('PARATERA_API_KEY', 'paratera-test-key-123456')
    ledger = BudgetLedger(tmp_path / 'budget.json', config.budget.hard_limit_usd)
    provider = ZhipuProvider(config, _selected(config), ledger, tmp_path / 'responses')
    payload = json.dumps({
        'id': 'server-response-7',
        'model': 'GLM-5.3',
        'choices': [{'message': {'content': 'done'}}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15},
    }).encode()

    class Response:
        status = 200
        headers = {'X-Request-Id': 'transport-request-9'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: Response())
    value, _ = provider.call_text(
        'experiment_code', 'diagnosed-success', 'secret system text', 'secret prompt text'
    )
    assert value == 'done'
    path = tmp_path / 'responses' / 'diagnostics' / 'diagnosed-success.json'
    raw_text = path.read_text(encoding='utf-8')
    diagnostic = json.loads(raw_text)
    assert diagnostic['state'] == 'completed'
    assert diagnostic['input_tokens'] == 12
    assert diagnostic['usage']['completion_tokens'] == 3
    assert diagnostic['attempts'][0]['http_status'] == 200
    assert diagnostic['attempts'][0]['server_request_id'] == 'transport-request-9'
    assert 'secret system text' not in raw_text
    assert 'secret prompt text' not in raw_text
    assert 'paratera-test-key' not in raw_text


def test_generic_parameter_400_is_retried_once_but_specific_error_is_not(
    project_root, tmp_path, monkeypatch
):
    import io
    import urllib.error

    config = _load(project_root)
    monkeypatch.setenv('PARATERA_API_KEY', 'paratera-test-key-123456')
    provider = ZhipuProvider(
        config, _selected(config),
        BudgetLedger(tmp_path / 'budget.json', config.budget.hard_limit_usd),
        tmp_path / 'responses',
    )
    calls = []

    def response_error(message):
        return urllib.error.HTTPError(
            'https://example.invalid', 400, 'Bad Request', {},
            io.BytesIO(json.dumps({'error': {'message': message}}).encode()),
        )

    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                'id': 'ok-after-route-retry', 'model': 'GLM-5.3',
                'choices': [{'message': {'content': 'done'}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
            }).encode()

    def transient(request, *args, **kwargs):
        calls.append(json.loads(request.data.decode("utf-8")))
        if len(calls) == 1:
            raise response_error('OpenAIException - A parameter specified in the request is not valid')
        return Response()

    monkeypatch.setattr('urllib.request.urlopen', transient)
    assert provider.call_text('experiment_code', 'route-retry', 's', 'p')[0] == 'done'
    assert len(calls) == 2
    assert calls[0]['max_tokens'] == 12000
    assert calls[1]['max_tokens'] == 6000
    assert calls[1]['messages'] == calls[0]['messages']

    calls.clear()
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            response_error('thinking.type disabled is not supported by this model')
        ),
    )
    with pytest.raises(ProviderError, match='thinking.type disabled'):
        provider.call_text('experiment_code', 'specific-error', 's', 'p')


def _provider(project_root, tmp_path, monkeypatch, responses):
    config = _load(project_root)
    monkeypatch.setenv("PARATERA_API_KEY", "paratera-test-key-123456")
    ledger = BudgetLedger(tmp_path / "budget.json", config.budget.hard_limit_usd)
    provider = ZhipuProvider(config, _selected(config), ledger, tmp_path / "responses")

    calls = []

    def fake_post(path, body, *args):
        calls.append({"path": path, "body": body})
        return responses.pop(0)

    monkeypatch.setattr(
        ZhipuProvider, "_post_json", lambda self, path, body, *args: fake_post(path, body, *args)
    )
    return provider, ledger, calls


def test_call_json_repairs_prose_response(project_root, tmp_path, monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    responses = [
        {
            "id": "resp-prose",
            "model": "GLM-5.3",
            "choices": [{"message": {"content": "抱歉，这个要求我需要更多背景信息才能回答。"}}],
            "usage": usage,
        },
        {
            "id": "resp-repair",
            "model": "GLM-5.3",
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
    assert config.provider.timeout_seconds == 600
    assert config.roles["experiment_code"].max_output_tokens == 12000
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
    assert model.context_length == 1048576


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


@pytest.mark.parametrize("model_id", ["GLM-5.3-Flash", "GLM-5.3"])
def test_glm53_reasoning_parameters_preserve_structured_calls(
    project_root, tmp_path, monkeypatch, model_id
):
    from dataclasses import replace

    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, [{
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 16, "completion_tokens": 3, "total_tokens": 19},
    }])
    # Synthetic pricing from the fixture, not a production price declaration.
    provider.selected_models["ideation"] = replace(
        provider.selected_models["ideation"], model_id=model_id
    )
    value, _ = provider.call_json(
        "ideation", "reasoning-test", "system", "prompt", "result", {"type": "object"}
    )
    assert value == {"ok": True}
    body = calls[0]["body"]
    assert "tools" in body
    assert body["reasoning_effort"] == "low"
    assert "thinking" not in body
    assert ledger.snapshot().reserved_usd == 0


@pytest.mark.parametrize("model_id", ["GLM-5.2", "GLM-5.1"])
def test_glm52_reasoning_is_disabled_with_legacy_parameter(
    project_root, tmp_path, monkeypatch, model_id
):
    from dataclasses import replace

    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, [{
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 16, "completion_tokens": 3, "total_tokens": 19},
    }])
    provider.selected_models["ideation"] = replace(
        provider.selected_models["ideation"], model_id=model_id
    )
    value, _ = provider.call_json(
        "ideation", "legacy-reasoning-test", "system", "prompt", "result",
        {"type": "object"},
    )
    assert value == {"ok": True}
    body = calls[0]["body"]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert ledger.snapshot().reserved_usd == 0


def test_call_json_tool_call_settles_and_caches(project_root, tmp_path, monkeypatch):
    responses = [
        {
            "id": "zhipu-1",
            "model": "GLM-5.3",
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
            "model": "GLM-5.3",
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

    def fake_post(path, body, *args):
        calls.append(dict(body))
        if len(calls) == 1:
            raise ProviderError(
                "Zhipu HTTP request failed with status 400: tools is not supported"
            )
        return {
            "id": "zhipu-4",
            "model": "GLM-5.3",
            "choices": [{"message": {"content": '{"Name": "n", "Title": "t"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(
        ZhipuProvider, "_post_json", lambda self, path, body, *args: fake_post(path, body, *args)
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
            "model": "GLM-5.3",
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


@pytest.mark.parametrize("invalid", [
    {"supported": True},
    {"supported": "true", "unsupported_reasons": []},
    {"supported": True, "unsupported_reasons": [], "unexpected": 1},
])
def test_schema_invalid_cached_json_is_retried_with_accounting(
    project_root, tmp_path, monkeypatch, invalid
):
    import json

    schema = {"type": "object", "properties": {
        "supported": {"type": "boolean"},
        "unsupported_reasons": {"type": "array", "items": {"type": "string"}},
    }, "required": ["supported", "unsupported_reasons"], "additionalProperties": False}
    valid = {"supported": True, "unsupported_reasons": []}
    responses = [{"choices": [{"message": {"content": json.dumps(value)}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
                 for value in (invalid, valid)]
    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, responses)
    # Reproduce an already saved malformed reply from before schema validation.
    provider._call("ideation", "schema-check", "system", "prompt", "contract", schema)
    value, metadata = provider.call_json(
        "ideation", "schema-check", "system", "prompt", "contract", schema
    )
    assert value == valid
    assert metadata["json_retry_attempt"] == 1
    assert len(calls) == 2
    assert "Validation failure" in calls[1]["body"]["messages"][1]["content"]
    original = json.loads((tmp_path / "responses/schema-check.json").read_text())
    assert original["value"] == invalid
    requests = json.loads(ledger.path.read_text())["requests"]
    assert len(requests) == 2
    assert all(row["state"] == "settled" for row in requests.values())
    provider.call_json("ideation", "schema-check", "system", "prompt", "contract", schema)
    assert len(calls) == 2


def test_schema_failures_exhaust_existing_limit_without_filling_fields(
    project_root, tmp_path, monkeypatch
):
    responses = [{"choices": [{"message": {"content": '{"supported": true}'}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
                 for _ in range(3)]
    provider, ledger, calls = _provider(project_root, tmp_path, monkeypatch, responses)
    with pytest.raises(ProviderError, match="unsupported_reasons"):
        provider.call_json("ideation", "missing", "system", "prompt", "contract",
                           {"type": "object", "required": ["unsupported_reasons"]})
    assert len(calls) == 3
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0


def test_empty_text_uses_stable_retry_and_cache(project_root, tmp_path, monkeypatch):
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    responses = [
        {
            "id": "empty-first",
            "model": "GLM-5.3",
            "choices": [{"message": {"content": ""}}],
            "usage": usage,
        },
        {
            "id": "retry-success",
            "model": "GLM-5.3",
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
    assert "reasoning_effort" not in calls[0]["body"]
    assert "reasoning_effort" not in calls[1]["body"]
    assert (tmp_path / "responses" / "paper-empty.json").is_file()
    assert (tmp_path / "responses" / "paper-empty-text-v2-retry-1.json").is_file()
    assert ledger.snapshot().spent_usd > 0
    cached_value, cached_metadata = provider.call_text(
        "paper_writer", "paper-empty", "system", "Write the paper"
    )
    assert cached_value == value
    assert cached_metadata["empty_retry_attempt"] == 1
    assert len(calls) == 2
