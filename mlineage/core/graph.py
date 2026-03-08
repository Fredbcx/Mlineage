"""
mlineage.core.graph
~~~~~~~~~~~~~~~~~~~
The LineageGraph manages the directed acyclic graph (DAG) of model versions.
It provides traversal, querying, and ancestry resolution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Generator, Optional

from mlineage.core.models import LineageEdge, ModelVersion


class LineageGraph:
    """
    A directed acyclic graph where nodes are ModelVersions and edges represent
    the relationships between them (updates, fine-tunes, etc.).

    The graph is stored as adjacency lists for efficient traversal in both
    directions (ancestors and descendants).
    """

    def __init__(self) -> None:
        self._versions: dict[str, ModelVersion] = {}
        self._edges: list[LineageEdge] = []
        # Adjacency: parent_id -> list of child_ids
        self._children: dict[str, list[str]] = defaultdict(list)
        # Adjacency: child_id -> list of parent_ids
        self._parents: dict[str, list[str]] = defaultdict(list)

    def add_version(self, version: ModelVersion) -> None:
        """Add a ModelVersion node to the graph."""
        if version.id in self._versions:
            raise ValueError(f"Version {version.id!r} already exists in the graph.")
        self._versions[version.id] = version

        for parent_id in version.parent_ids:
            edge = LineageEdge(source_id=parent_id, target_id=version.id)
            self.add_edge(edge)

    def add_edge(self, edge: LineageEdge) -> None:
        """Add a directed edge between two existing version nodes."""
        if edge.source_id not in self._versions:
            raise KeyError(f"Source version {edge.source_id!r} not found in graph.")
        if edge.target_id not in self._versions:
            raise KeyError(f"Target version {edge.target_id!r} not found in graph.")

        self._edges.append(edge)
        self._children[edge.source_id].append(edge.target_id)
        self._parents[edge.target_id].append(edge.source_id)

    def get_version(self, version_id: str) -> ModelVersion:
        """Retrieve a version by its ID."""
        try:
            return self._versions[version_id]
        except KeyError:
            raise KeyError(f"Version {version_id!r} not found in graph.")

    def roots(self) -> list[ModelVersion]:
        """Return all root nodes (versions with no parents)."""
        return [v for v in self._versions.values() if v.is_root()]

    def leaves(self) -> list[ModelVersion]:
        """Return all leaf nodes (versions with no children)."""
        return [
            v for vid, v in self._versions.items()
            if not self._children.get(vid)
        ]

    def ancestors(
        self, version_id: str, max_depth: Optional[int] = None
    ) -> Generator[ModelVersion, None, None]:
        """
        Yield all ancestor versions of the given version in breadth-first order.
        Optionally limit to ``max_depth`` hops.
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(version_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited or current_id == version_id:
                visited.add(current_id)
                for parent_id in self._parents.get(current_id, []):
                    if parent_id not in visited:
                        if max_depth is None or depth < max_depth:
                            queue.append((parent_id, depth + 1))
                continue

            visited.add(current_id)
            yield self._versions[current_id]

            for parent_id in self._parents.get(current_id, []):
                if parent_id not in visited:
                    if max_depth is None or depth < max_depth:
                        queue.append((parent_id, depth + 1))

    def descendants(
        self, version_id: str, max_depth: Optional[int] = None
    ) -> Generator[ModelVersion, None, None]:
        """
        Yield all descendant versions of the given version in breadth-first order.
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(version_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited or current_id == version_id:
                visited.add(current_id)
                for child_id in self._children.get(current_id, []):
                    if child_id not in visited:
                        if max_depth is None or depth < max_depth:
                            queue.append((child_id, depth + 1))
                continue

            visited.add(current_id)
            yield self._versions[current_id]

            for child_id in self._children.get(current_id, []):
                if child_id not in visited:
                    if max_depth is None or depth < max_depth:
                        queue.append((child_id, depth + 1))

    def history(self, version_id: str) -> list[ModelVersion]:
        """
        Return the full linear history from the root to this version.
        Only well-defined for linear chains; for branched graphs, use ancestors().
        """
        chain: list[ModelVersion] = []
        current_id: Optional[str] = version_id

        while current_id is not None:
            version = self._versions[current_id]
            chain.append(version)
            parents = self._parents.get(current_id, [])
            current_id = parents[0] if parents else None

        chain.reverse()
        return chain

    def all_versions(self) -> list[ModelVersion]:
        """Return all versions in the graph, sorted by creation time."""
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def __len__(self) -> int:
        return len(self._versions)

    def __repr__(self) -> str:
        return (
            f"LineageGraph(versions={len(self._versions)}, "
            f"edges={len(self._edges)})"
        )
