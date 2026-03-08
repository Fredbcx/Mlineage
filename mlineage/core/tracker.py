"""
mlineage.core.tracker
~~~~~~~~~~~~~~~~~~~~~
The Tracker is the main entry point for MLineage. It manages the lifecycle
of model versions: logging, committing, querying, and reproducing.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from mlineage.core.graph import LineageGraph
from mlineage.core.models import DataSnapshot, ModelVersion, VersionStatus


class VersionLogger:
    """
    Context manager returned by ``Tracker.log_version()``.
    Accumulates version data during the training run and commits on exit.

    Example::

        with tracker.log_version(parent_id=prev_id) as v:
            v.model_path = "./checkpoints/model_latest.pt"
            v.dataset = DataSnapshot.from_path("./data/feb_2024/")
            v.metrics = {"accuracy": 0.95}
            v.notes = "February update"
        # version is committed here
    """

    def __init__(self, tracker: "Tracker", parent_id: Optional[str] = None) -> None:
        self._tracker = tracker
        self._parent_ids: list[str] = [parent_id] if parent_id else []

        # Public attributes set by user during context
        self.model_path: Optional[str] = None
        self.dataset: Optional[DataSnapshot] = None
        self.metrics: dict[str, float] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.notes: Optional[str] = None
        self.tags: list[str] = []
        self.metadata: dict[str, Any] = {}

        self._version: Optional[ModelVersion] = None

    def __enter__(self) -> "VersionLogger":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        if exc_type is not None:
            # Don't commit if an exception occurred during training
            return False

        if self.model_path is None:
            raise ValueError("model_path must be set before exiting the context.")
        if self.dataset is None:
            raise ValueError("dataset must be set before exiting the context.")

        version = ModelVersion(
            model_name=self._tracker.model_name,
            model_path=self.model_path,
            dataset=self.dataset,
            parent_ids=self._parent_ids,
            metrics=self.metrics,
            hyperparameters=self.hyperparameters,
            notes=self.notes,
            tags=self.tags,
            metadata=self.metadata,
        )
        version.commit()
        self._version = version
        self._tracker._register_version(version)
        return False

    @property
    def version(self) -> Optional[ModelVersion]:
        """The committed ModelVersion, available after exiting the context."""
        return self._version


class Tracker:
    """
    Tracks the lineage of a single named model across multiple versions.

    Example::

        tracker = Tracker("fraud-detector", storage="./lineage-store")

        with tracker.log_version() as v:
            v.model_path = "./checkpoints/model_v1.pt"
            v.dataset = DataSnapshot.from_path("./data/jan_2024/")
            v.metrics = {"accuracy": 0.92}

        with tracker.log_version(parent_id=v.version.id) as v2:
            v2.model_path = "./checkpoints/model_v2.pt"
            v2.dataset = DataSnapshot.from_path("./data/feb_2024/")
            v2.metrics = {"accuracy": 0.95}

        for version in tracker.history():
            print(version)
    """

    def __init__(self, model_name: str, storage: str = ".mlineage") -> None:
        self.model_name = model_name
        self.storage_path = storage
        self._graph = LineageGraph()

    @contextlib.contextmanager
    def log_version(
        self, parent_id: Optional[str] = None
    ) -> Generator[VersionLogger, None, None]:
        """
        Context manager to log a new model version.

        If ``parent_id`` is not provided and there are existing versions,
        the most recently committed version is used as the parent.

        Example::

            with tracker.log_version() as v:
                v.model_path = "..."
                v.dataset = DataSnapshot(...)
                v.metrics = {"accuracy": 0.94}
        """
        if parent_id is None:
            latest = self.latest()
            parent_id = latest.id if latest is not None else None

        logger = VersionLogger(self, parent_id=parent_id)
        try:
            yield logger
        except Exception:
            raise
        else:
            if logger.model_path is None:
                raise ValueError("model_path must be set before exiting the context.")
            if logger.dataset is None:
                raise ValueError("dataset must be set before exiting the context.")
            version = ModelVersion(
                model_name=self.model_name,
                model_path=logger.model_path,
                dataset=logger.dataset,
                parent_ids=logger._parent_ids,
                metrics=logger.metrics,
                hyperparameters=logger.hyperparameters,
                notes=logger.notes,
                tags=logger.tags,
                metadata=logger.metadata,
            )
            version.commit()
            logger._version = version
            self._register_version(version)

    def _register_version(self, version: ModelVersion) -> None:
        """Internal: add a committed version to the graph."""
        self._graph.add_version(version)

    def latest(self) -> Optional[ModelVersion]:
        """Return the most recently committed version, or None if no versions exist."""
        committed = [
            v for v in self._graph.all_versions()
            if v.status == VersionStatus.COMMITTED
        ]
        if not committed:
            return None
        return max(committed, key=lambda v: v.committed_at or v.created_at)

    def history(self, from_version_id: Optional[str] = None) -> list[ModelVersion]:
        """
        Return the linear version history, newest first.

        If ``from_version_id`` is provided, returns the history up to that version.
        Otherwise returns the full history from the latest version.
        """
        version_id = from_version_id
        if version_id is None:
            latest = self.latest()
            if latest is None:
                return []
            version_id = latest.id

        return list(reversed(self._graph.history(version_id)))

    def get_version(self, version_id: str) -> ModelVersion:
        """Retrieve a specific version by its ID."""
        return self._graph.get_version(version_id)

    def blame(
        self, metric: str, direction: str = "decrease"
    ) -> Optional[ModelVersion]:
        """
        Find the version that caused a metric to change significantly.

        ``direction`` can be "decrease" (to find regressions) or "increase".

        Returns the version where the largest change occurred, or None
        if no history is available.

        Note: this is a naive implementation — a proper blame algorithm
        will be added in a future release.
        """
        versions = self.history()
        if len(versions) < 2:
            return None

        sign = -1 if direction == "decrease" else 1
        worst_delta: float = 0.0
        worst_version: Optional[ModelVersion] = None

        for i in range(1, len(versions)):
            prev = versions[i - 1]
            curr = versions[i]
            delta = curr.metric_delta(prev, metric)
            if delta is None:
                continue
            signed_delta = sign * delta
            if signed_delta < worst_delta:
                worst_delta = signed_delta
                worst_version = curr

        return worst_version

    def summary(self) -> str:
        """Return a human-readable summary of the tracked model's history."""
        versions = self.history()
        if not versions:
            return f"Tracker({self.model_name!r}): no versions recorded."

        lines = [f"Model: {self.model_name}", f"Versions: {len(versions)}", ""]
        for v in versions:
            committed = (
                v.committed_at.strftime("%Y-%m-%d %H:%M") if v.committed_at else "N/A"
            )
            metrics_str = ", ".join(f"{k}={val:.4f}" for k, val in v.metrics.items())
            lines.append(f"  [{committed}] {v.id[:8]}...  {metrics_str}")
            if v.notes:
                lines.append(f"             {v.notes}")

        return "\n".join(lines)

    @classmethod
    def load(cls, model_name: str, storage: str = ".mlineage") -> "Tracker":
        """
        Load an existing tracker from storage.

        Note: persistence is not yet implemented — this currently returns
        a fresh tracker. Storage backends are on the roadmap.
        """
        # TODO: deserialize from storage backend
        return cls(model_name=model_name, storage=storage)

    def __repr__(self) -> str:
        return (
            f"Tracker(model={self.model_name!r}, "
            f"versions={len(self._graph)}, "
            f"storage={self.storage_path!r})"
        )
