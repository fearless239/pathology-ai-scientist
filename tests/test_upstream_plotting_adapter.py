import pytest

from pathmnist.upstream_adapter import UpstreamIntegrationError, UpstreamPlottingAdapter


def test_optional_upstream_plot_code_rejects_network_modules(tmp_path):
    adapter = UpstreamPlottingAdapter(tmp_path)
    with pytest.raises(UpstreamIntegrationError, match="network"):
        adapter.validate_code("import requests\nrequests.get('https://example.com')")


def test_optional_upstream_plot_code_accepts_basic_matplotlib(tmp_path):
    adapter = UpstreamPlottingAdapter(tmp_path)
    adapter.validate_code("import matplotlib.pyplot as plt\nplt.plot([1], [2])")
