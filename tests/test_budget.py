import json

import pytest

from gate_a.budget import BudgetExceeded, BudgetLedger, LedgerError


def test_reservation_settlement_and_release(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    assert ledger.reserve("request-a", 0.7, {"role": "test"})
    assert ledger.snapshot().reserved_usd == 0.7

    ledger.settle("request-a", 0.25, {"prompt_tokens": 10})
    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == 0.25
    assert snapshot.reserved_usd == 0
    assert snapshot.available_usd == 1.75
    assert ledger.reserve("request-a", 0.7, {"role": "test"}) is False

    assert ledger.reserve("request-b", 0.5, {"role": "test"})
    ledger.release("request-b", "offline failure")
    assert ledger.snapshot().available_usd == 1.75


def test_durable_response_recovers_ledger_settlement_once(tmp_path):
    ledger = BudgetLedger(tmp_path / 'budget.json', 2.0)
    ledger.reserve('request', 0.7, {})
    response = {'actual_cost_usd': 0.2, 'usage': {'prompt_tokens': 5}}
    ledger.recover_cached('request', response)
    ledger.recover_cached('request', response)
    assert ledger.snapshot().spent_usd == 0.2
    assert ledger.snapshot().reserved_usd == 0


def test_released_request_can_be_reserved_again(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    assert ledger.reserve("request-retry", 0.5, {"role": "test"})
    ledger.release("request-retry", "transient provider failure")
    assert ledger.reserve("request-retry", 0.5, {"role": "test"}) is True
    ledger.settle("request-retry", 0.2, {"prompt_tokens": 5})
    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == 0.2
    assert snapshot.reserved_usd == 0.0


def test_hard_limit_is_checked_before_request(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    ledger.reserve("request-a", 1.6, {})
    with pytest.raises(BudgetExceeded):
        ledger.reserve("request-b", 0.5, {})


def test_actual_cost_cannot_exceed_reservation(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", 2.0)
    ledger.reserve("request-a", 0.2, {})
    with pytest.raises(LedgerError):
        ledger.settle("request-a", 0.21, {})
    raw = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert raw["requests"]["request-a"]["state"] == "reserved"


def test_full_mode_upgrade_preserves_settled_requests(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path, 2.0)
    ledger.reserve("old", 0.5, {})
    ledger.settle("old", 0.25, {})
    upgraded = BudgetLedger.open_or_upgrade(path, 8.0)
    assert upgraded.snapshot().hard_limit_usd == 8.0
    assert upgraded.snapshot().spent_usd == 0.25
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["requests"]["old"]["state"] == "settled"
    assert raw["limit_history"][-1]["previous_usd"] == 2.0


def test_budget_upgrade_is_monotonic(tmp_path):
    path = tmp_path / "budget.json"
    BudgetLedger(path, 8.0)
    with pytest.raises(LedgerError, match="decreased"):
        BudgetLedger.open_or_upgrade(path, 2.0)
