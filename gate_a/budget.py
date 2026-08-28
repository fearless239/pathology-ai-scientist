from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised before a request that cannot be safely reserved."""


class LedgerError(RuntimeError):
    """Raised when the persistent ledger is internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class BudgetSnapshot:
    hard_limit_usd: float
    spent_usd: float
    reserved_usd: float
    available_usd: float


class BudgetLedger:
    """Small persistent ledger with request-level reservation and settlement.

    A request ID is idempotent. Settled requests cannot be reserved again, which prevents a
    resumed stage from silently paying twice.
    """

    def __init__(self, path: Path, hard_limit_usd: float):
        self.path = path
        self.hard_limit_usd = round(float(hard_limit_usd), 10)
        self._lock = threading.RLock()
        if not path.exists():
            _atomic_json_write(
                path,
                {
                    "schema_version": 1,
                    "hard_limit_usd": self.hard_limit_usd,
                    "requests": {},
                },
            )
        self._read()

    def _read(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if float(data["hard_limit_usd"]) != self.hard_limit_usd:
            raise LedgerError("Configured hard limit differs from the existing ledger")
        return data

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            data = self._read()
            spent = sum(
                float(item.get("actual_usd", 0))
                for item in data["requests"].values()
                if item["state"] == "settled"
            )
            reserved = sum(
                float(item["reserved_usd"])
                for item in data["requests"].values()
                if item["state"] == "reserved"
            )
            available = max(0.0, self.hard_limit_usd - spent - reserved)
            return BudgetSnapshot(
                hard_limit_usd=self.hard_limit_usd,
                spent_usd=round(spent, 10),
                reserved_usd=round(reserved, 10),
                available_usd=round(available, 10),
            )

    def recover_cached(self, request_id: str, metadata: dict[str, Any]) -> None:
        """Settle a durable response after a crash before ledger commit."""
        with self._lock:
            item = self._read()['requests'].get(request_id)
            if item and item['state'] == 'reserved':
                self.settle(request_id, metadata['actual_cost_usd'], metadata['usage'])

    @classmethod
    def open_or_upgrade(cls, path: Path, hard_limit_usd: float) -> "BudgetLedger":
        """Open a ledger and permit only an explicit monotonic limit increase.

        Settled and reserved request records are preserved byte-for-byte at the
        request level.  Archived callers should never invoke this helper.
        """
        requested = round(float(hard_limit_usd), 10)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            current = round(float(data["hard_limit_usd"]), 10)
            if requested < current:
                raise LedgerError("Budget hard limit cannot be decreased")
            if requested > current:
                data["hard_limit_usd"] = requested
                history = data.setdefault("limit_history", [])
                history.append({"previous_usd": current, "new_usd": requested, "changed_at": _utc_now()})
                _atomic_json_write(path, data)
        return cls(path, requested)

    def reserve(
        self, request_id: str, amount_usd: float, metadata: dict[str, Any]
    ) -> bool:
        amount = round(float(amount_usd), 10)
        if amount <= 0:
            raise LedgerError("Reservation must be positive")
        with self._lock:
            data = self._read()
            existing = data["requests"].get(request_id)
            if existing and existing["state"] != "released":
                if existing["state"] == "settled":
                    return False
                if (
                    existing["state"] == "reserved"
                    and float(existing["reserved_usd"]) == amount
                ):
                    return False
                raise LedgerError(
                    f"Request ID {request_id!r} already exists in state {existing['state']}"
                )

            snap = self.snapshot()
            if amount > snap.available_usd + 1e-10:
                raise BudgetExceeded(
                    f"Request {request_id!r} needs ${amount:.6f}, only ${snap.available_usd:.6f} remains"
                )
            data["requests"][request_id] = {
                "state": "reserved",
                "reserved_usd": amount,
                "actual_usd": None,
                "created_at": _utc_now(),
                "metadata": metadata,
            }
            _atomic_json_write(self.path, data)
            return True

    def settle(self, request_id: str, actual_usd: float, usage: dict[str, Any]) -> None:
        actual = round(float(actual_usd), 10)
        with self._lock:
            data = self._read()
            item = data["requests"].get(request_id)
            if not item or item["state"] != "reserved":
                raise LedgerError(f"Cannot settle unreserved request {request_id!r}")
            if actual < 0 or actual > float(item["reserved_usd"]) + 1e-8:
                raise LedgerError(
                    f"Actual cost ${actual:.8f} exceeds reservation ${item['reserved_usd']:.8f}"
                )
            item.update(
                state="settled",
                actual_usd=actual,
                settled_at=_utc_now(),
                usage=usage,
            )
            _atomic_json_write(self.path, data)

    def release(self, request_id: str, reason: str) -> None:
        with self._lock:
            data = self._read()
            item = data["requests"].get(request_id)
            if not item or item["state"] != "reserved":
                return
            item.update(
                state="released", released_at=_utc_now(), release_reason=reason[:500]
            )
            _atomic_json_write(self.path, data)
