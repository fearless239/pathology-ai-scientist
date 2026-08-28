"""Public extension contracts for Path-AI Scientist.

These protocols describe the beta extension boundary.  PathMNIST is the reference
implementation; third-party adapters should depend on these shapes instead of UI code.
The API may still change before 1.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .dataset_adapter import DatasetSpec
else:
    DatasetSpec = Any


@dataclass(frozen=True)
class ModelRoles:
    planner: str = "offline-fixture"
    experimenter: str = "offline-fixture"
    writer: str = "offline-fixture"
    reviewer: str = "offline-fixture-reviewer"


@dataclass(frozen=True)
class RunPermissions:
    allow_paid_llm: bool = False
    allow_gpu: bool = False
    allow_sealed_test: bool = False
    allow_pdf: bool = False


@dataclass(frozen=True)
class ResearchTaskConfig:
    """Provider-neutral configuration passed into the research workflow."""

    task_id: str
    direction: str
    dataset_adapter: str
    dataset_path: Path
    output_root: Path
    budget_usd: float = 0.0
    seed: int = 7
    model_roles: ModelRoles = field(default_factory=ModelRoles)
    permissions: RunPermissions = field(default_factory=RunPermissions)

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.direction.strip():
            raise ValueError("task_id and direction are required")
        if self.budget_usd < 0:
            raise ValueError("budget_usd cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset_path"] = str(self.dataset_path)
        value["output_root"] = str(self.output_root)
        return value


@runtime_checkable
class DatasetAdapter(Protocol):
    """Discovers data, fingerprints it, and enforces split isolation."""

    def discover(self, source: Path, profile_path: Path | None = None) -> DatasetSpec: ...


@runtime_checkable
class ExperimentBackend(Protocol):
    """Runs generated experiments behind a sandbox boundary."""

    def preflight(self, config: ResearchTaskConfig, dataset: DatasetSpec) -> Mapping[str, Any]: ...

    def run_experiments(
        self, config: ResearchTaskConfig, dataset: DatasetSpec
    ) -> Sequence[Mapping[str, Any]]: ...

    def freeze_candidate(
        self, config: ResearchTaskConfig, experiments: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...

    def evaluate_sealed_test(
        self, config: ResearchTaskConfig, candidate: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class ArtifactValidator(Protocol):
    """Validates that published claims resolve to immutable evidence."""

    def validate_manifest(self, task_root: Path) -> Mapping[str, Any]: ...

    def validate_metrics(self, task_root: Path) -> Mapping[str, Any]: ...

    def validate_publication(self, task_root: Path) -> Mapping[str, Any]: ...


__all__ = [
    "ArtifactValidator",
    "DatasetAdapter",
    "ExperimentBackend",
    "ModelRoles",
    "ResearchTaskConfig",
    "RunPermissions",
]
