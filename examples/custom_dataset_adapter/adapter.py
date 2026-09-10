"""Copyable trusted adapter template for supervised image classification."""

from pathlib import Path

from path_ai_scientist.adapters import DatasetAdapter, DatasetDiscoveryError, DatasetSpec


class CustomDatasetAdapter(DatasetAdapter):
    """Adapt a project-specific folder or manifest while retaining safety checks."""

    def discover(self, source: Path, profile_path: Path | None = None) -> DatasetSpec:
        spec = super().discover(source, profile_path=None)
        if len(spec.classes) != 2:
            raise DatasetDiscoveryError("This example adapter requires binary classification")
        spec.recommended_metrics = ["macro_f1", "accuracy", "weighted_f1"]
        if profile_path is not None:
            spec.write_profile(profile_path)
        return spec
