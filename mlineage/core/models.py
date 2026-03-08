"""
mlineage.core.models
~~~~~~~~~~~~~~~~~~~~
Core data models for MLineage. These are the fundamental building blocks
that represent the lineage graph: model versions, data snapshots, and
the relationships between them.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    HUGGINGFACE = "huggingface"


class VersionStatus(str, Enum):
    STAGED = "staged"       # Being logged, not yet committed
    COMMITTED = "committed" # Fully recorded
    ARCHIVED = "archived"   # Older version, retained for lineage
    DEPRECATED = "deprecated"


@dataclass
class DataSnapshot:
    """
    Represents an immutable snapshot of the data used to train or update
    a model version. The hash uniquely identifies the content.

    Example::

        snapshot = DataSnapshot(
            path="s3://my-bucket/data/jan_2024/",
            hash="sha256:abc123...",
            record_count=50_000,
            notes="January fraud patterns, post-holiday spike included",
        )
    """

    path: str
    hash: str
    record_count: Optional[int] = None
    schema_version: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path, notes: Optional[str] = None) -> "DataSnapshot":
        """
        Create a DataSnapshot by hashing the content at the given path.
        For directories, hashes the sorted list of file hashes.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        content_hash = _hash_path(p)
        return cls(path=str(path), hash=content_hash, notes=notes)

    def __str__(self) -> str:
        return f"DataSnapshot(path={self.path!r}, hash={self.hash[:16]}...)"


@dataclass
class EnvironmentSnapshot:
    """
    Captures the computational environment at the time of a training run.
    Used for reproducibility — allows reconstructing the exact environment
    from a version's lineage record.
    """

    python_version: str
    dependencies: dict[str, str]  # package_name -> version
    cuda_version: Optional[str] = None
    platform: Optional[str] = None
    environment_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.environment_hash is None:
            self.environment_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.python_version}:{sorted(self.dependencies.items())}"
        return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ModelVersion:
    """
    A single node in the lineage graph — one version of a model at a point
    in time, with full context about how it was created.

    Versions are connected to their parents via ``parent_ids``, forming
    the directed acyclic graph (DAG) that MLineage tracks.

    Example::

        version = ModelVersion(
            model_name="fraud-detector",
            model_path="./checkpoints/model_20240201.pt",
            dataset=DataSnapshot.from_path("./data/jan_2024/"),
            metrics={"accuracy": 0.94, "f1": 0.91},
            hyperparameters={"lr": 1e-4, "epochs": 3},
            notes="Monthly update with January fraud patterns",
        )
    """

    model_name: str
    model_path: str
    dataset: DataSnapshot

    # Lineage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: list[str] = field(default_factory=list)

    # Training context
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    environment: Optional[EnvironmentSnapshot] = None

    # Metadata
    status: VersionStatus = VersionStatus.STAGED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    committed_at: Optional[datetime] = None
    notes: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def commit(self) -> None:
        """Mark this version as fully recorded."""
        self.status = VersionStatus.COMMITTED
        self.committed_at = datetime.now(timezone.utc)

    def is_root(self) -> bool:
        """True if this version has no parents (base model or initial training)."""
        return len(self.parent_ids) == 0

    def metric_delta(self, other: "ModelVersion", metric: str) -> Optional[float]:
        """
        Compute the change in a metric relative to another version.
        Returns None if the metric is not present in both versions.
        """
        if metric not in self.metrics or metric not in other.metrics:
            return None
        return self.metrics[metric] - other.metrics[metric]

    def __str__(self) -> str:
        parents = f", parents={self.parent_ids}" if self.parent_ids else ""
        return (
            f"ModelVersion(id={self.id[:8]}..., "
            f"model={self.model_name!r}, "
            f"status={self.status.value}"
            f"{parents})"
        )


@dataclass
class LineageEdge:
    """
    A directed edge in the lineage DAG connecting two ModelVersions.
    ``source_id`` is the parent (earlier) version; ``target_id`` is the child.
    """

    source_id: str
    target_id: str
    edge_type: str = "update"  # "update", "fine_tune", "distill", "merge"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_path(path: Path) -> str:
    """Compute a deterministic SHA-256 hash for a file or directory."""
    h = hashlib.sha256()

    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                h.update(str(child.relative_to(path)).encode())
                with open(child, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
    else:
        raise ValueError(f"Path is neither a file nor a directory: {path}")

    return "sha256:" + h.hexdigest()
