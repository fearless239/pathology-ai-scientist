import hashlib

import pytest

from gate_a.provider import ProviderError, _safe_request_id


def test_long_safe_request_id_is_deterministically_compacted():
    original = "a" * 95 + "-json-v2-retry-2"
    expected = original[:75] + "-" + hashlib.sha256(original.encode()).hexdigest()[:24]

    assert _safe_request_id(original) == expected
    assert len(expected) == 100


def test_request_id_still_rejects_unsafe_characters():
    with pytest.raises(ProviderError, match="Unsafe request ID"):
        _safe_request_id("unsafe/request")
