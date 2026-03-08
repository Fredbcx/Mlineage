"""Tests for mlineage.core.models and mlineage.core.tracker."""

import pytest
from datetime import datetime, timezone

import mlineage as ml
from mlineage.core.models import DataSnapshot, ModelVersion, VersionStatus
from mlineage.core.graph import LineageGraph
from mlineage.core.tracker import Tracker


# ---------------------------------------------------------------------------
# DataSnapshot
# ---------------------------------------------------------------------------

class TestDataSnapshot:
    def test_basic_creation(self) -> None:
        snap = DataSnapshot(path="s3://bucket/data/", hash="sha256:abc123")
        assert snap.path == "s3://bucket/data/"
        assert snap.hash == "sha256:abc123"
        assert snap.record_count is None

    def test_with_metadata(self) -> None:
        snap = DataSnapshot(
            path="./data/",
            hash="sha256:abc",
            record_count=10_000,
            notes="January dataset",
        )
        assert snap.record_count == 10_000
        assert snap.notes == "January dataset"

    def test_str_repr(self) -> None:
        snap = DataSnapshot(path="./data/", hash="sha256:" + "a" * 64)
        s = str(snap)
        assert "DataSnapshot" in s
        assert "./data/" in s


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------

class TestModelVersion:
    def _make_snapshot(self) -> DataSnapshot:
        return DataSnapshot(path="./data/", hash="sha256:abc")

    def test_basic_creation(self) -> None:
        v = ModelVersion(
            model_name="my-model",
            model_path="./model.pt",
            dataset=self._make_snapshot(),
        )
        assert v.model_name == "my-model"
        assert v.status == VersionStatus.STAGED
        assert v.is_root()

    def test_commit(self) -> None:
        v = ModelVersion(
            model_name="my-model",
            model_path="./model.pt",
            dataset=self._make_snapshot(),
        )
        assert v.committed_at is None
        v.commit()
        assert v.status == VersionStatus.COMMITTED
        assert v.committed_at is not None

    def test_metric_delta(self) -> None:
        snap = self._make_snapshot()
        v1 = ModelVersion("m", "./v1.pt", snap, metrics={"acc": 0.90})
        v2 = ModelVersion("m", "./v2.pt", snap, metrics={"acc": 0.95})
        delta = v2.metric_delta(v1, "acc")
        assert delta == pytest.approx(0.05)

    def test_metric_delta_missing_key(self) -> None:
        snap = self._make_snapshot()
        v1 = ModelVersion("m", "./v1.pt", snap, metrics={"acc": 0.90})
        v2 = ModelVersion("m", "./v2.pt", snap, metrics={"f1": 0.88})
        assert v2.metric_delta(v1, "acc") is None

    def test_not_root_when_has_parent(self) -> None:
        snap = self._make_snapshot()
        v = ModelVersion("m", "./v.pt", snap, parent_ids=["parent-id"])
        assert not v.is_root()


# ---------------------------------------------------------------------------
# LineageGraph
# ---------------------------------------------------------------------------

class TestLineageGraph:
    def _make_version(self, name: str = "m", parent_ids: list[str] | None = None) -> ModelVersion:
        snap = DataSnapshot(path="./data/", hash="sha256:abc")
        v = ModelVersion(name, f"./{name}.pt", snap, parent_ids=parent_ids or [])
        v.commit()
        return v

    def test_add_and_retrieve(self) -> None:
        g = LineageGraph()
        v = self._make_version()
        g.add_version(v)
        assert g.get_version(v.id) is v

    def test_duplicate_version_raises(self) -> None:
        g = LineageGraph()
        v = self._make_version()
        g.add_version(v)
        with pytest.raises(ValueError):
            g.add_version(v)

    def test_roots_and_leaves(self) -> None:
        g = LineageGraph()
        v1 = self._make_version("v1")
        g.add_version(v1)
        v2 = self._make_version("v2", parent_ids=[v1.id])
        g.add_version(v2)

        roots = g.roots()
        leaves = g.leaves()
        assert len(roots) == 1 and roots[0].id == v1.id
        assert len(leaves) == 1 and leaves[0].id == v2.id

    def test_ancestors(self) -> None:
        g = LineageGraph()
        v1 = self._make_version("v1")
        g.add_version(v1)
        v2 = self._make_version("v2", parent_ids=[v1.id])
        g.add_version(v2)
        v3 = self._make_version("v3", parent_ids=[v2.id])
        g.add_version(v3)

        ancestor_ids = {v.id for v in g.ancestors(v3.id)}
        assert v1.id in ancestor_ids
        assert v2.id in ancestor_ids
        assert v3.id not in ancestor_ids

    def test_history_linear(self) -> None:
        g = LineageGraph()
        v1 = self._make_version("v1")
        g.add_version(v1)
        v2 = self._make_version("v2", parent_ids=[v1.id])
        g.add_version(v2)
        v3 = self._make_version("v3", parent_ids=[v2.id])
        g.add_version(v3)

        history = g.history(v3.id)
        assert [v.id for v in history] == [v1.id, v2.id, v3.id]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TestTracker:
    def test_empty_tracker(self) -> None:
        t = Tracker("test-model")
        assert t.latest() is None
        assert t.history() == []

    def test_log_single_version(self) -> None:
        t = Tracker("test-model")
        with t.log_version() as v:
            v.model_path = "./model.pt"
            v.dataset = DataSnapshot(path="./data/", hash="sha256:abc")
            v.metrics = {"accuracy": 0.90}

        assert v.version is not None
        assert v.version.status == VersionStatus.COMMITTED
        assert t.latest() is not None

    def test_log_multiple_versions_auto_parent(self) -> None:
        t = Tracker("test-model")

        with t.log_version() as v1:
            v1.model_path = "./model_v1.pt"
            v1.dataset = DataSnapshot(path="./data_v1/", hash="sha256:aaa")
            v1.metrics = {"accuracy": 0.90}

        with t.log_version() as v2:
            v2.model_path = "./model_v2.pt"
            v2.dataset = DataSnapshot(path="./data_v2/", hash="sha256:bbb")
            v2.metrics = {"accuracy": 0.93}

        assert v2.version is not None
        assert v1.version is not None
        assert v1.version.id in v2.version.parent_ids

    def test_history_ordering(self) -> None:
        t = Tracker("test-model")
        for i in range(3):
            with t.log_version() as v:
                v.model_path = f"./model_v{i}.pt"
                v.dataset = DataSnapshot(path=f"./data_{i}/", hash=f"sha256:{i:064d}")
                v.metrics = {"accuracy": 0.90 + i * 0.02}

        history = t.history()
        assert len(history) == 3
        # history() returns newest first
        assert history[0].metrics["accuracy"] > history[-1].metrics["accuracy"]

    def test_missing_model_path_raises(self) -> None:
        t = Tracker("test-model")
        with pytest.raises(ValueError, match="model_path"):
            with t.log_version() as v:
                v.dataset = DataSnapshot(path="./data/", hash="sha256:abc")
                # model_path intentionally not set

    def test_missing_dataset_raises(self) -> None:
        t = Tracker("test-model")
        with pytest.raises(ValueError, match="dataset"):
            with t.log_version() as v:
                v.model_path = "./model.pt"
                # dataset intentionally not set

    def test_summary_output(self) -> None:
        t = Tracker("fraud-detector")
        with t.log_version() as v:
            v.model_path = "./model.pt"
            v.dataset = DataSnapshot(path="./data/", hash="sha256:abc")
            v.metrics = {"accuracy": 0.94}
            v.notes = "Initial version"

        summary = t.summary()
        assert "fraud-detector" in summary
        assert "0.9400" in summary
        assert "Initial version" in summary
