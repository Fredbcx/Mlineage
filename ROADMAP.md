# MLineage Roadmap

This document describes the planned development trajectory of MLineage.
It's a living document — priorities may shift based on community feedback.

The project follows a four-layer architecture built incrementally:
**Tracking → Analysis → Visualization → Reproducibility**

Each layer is useful on its own. You don't need the full stack to get value.

---

## Layer 1 — Tracking Core *(in progress)*

The foundation. Without this, nothing else works.

### ✅ Done
- `ModelVersion`, `DataSnapshot`, `LineageEdge` data models with strict mypy typing
- `LineageGraph` — DAG with BFS-based ancestor/descendant traversal
- `Tracker` API with context-manager interface (`log_version`)
- `blame()` — naive regression detection across metric history
- 20 unit tests passing

### 🔲 Next
- **Local storage backend** — persist the lineage graph to disk (JSON or SQLite) so history survives across sessions. This is the most important missing piece.
- **`Tracker.load()`** — fully functional load from persisted storage
- **CLI** — `mlineage history <model>`, `mlineage blame <model> --metric accuracy`

---

## Layer 2 — Analysis *(planned)*

Makes the lineage graph analytically useful, not just structural.

### Data drift detection
Connect to [Evidently](https://github.com/evidentlyai/evidently) or [alibi-detect](https://github.com/SeldonIO/alibi-detect) to compute, per `DataSnapshot` transition:
- Population Stability Index (PSI) per feature
- KL divergence / Jensen-Shannon divergence
- Kolmogorov-Smirnov test for continuous features
- Label distribution shift

Each drift report is stored as metadata on the `LineageEdge` connecting two versions.

### Model behavioral fingerprinting
Track how the model responds to a fixed "probe set" — a curated set of inputs chosen to cover edge cases and critical subgroups.

```python
fingerprint = ml.Fingerprint.from_probe_set("./probes/fraud_edge_cases.csv")
tracker.register_fingerprint(fingerprint)

# After each update, behavioral shift is recorded automatically
with tracker.log_version() as v:
    ...
    v.fingerprint = fingerprint.evaluate(model)
```

### Embedding drift
For neural models, measure representational shift between versions:
- Cosine similarity between mean embeddings on a reference dataset
- CKA (Centered Kernel Alignment) for deeper comparison
- t-SNE/UMAP snapshots stored per version

### Confidence calibration tracking
- Expected Calibration Error (ECE) per version
- Reliability diagram data stored in version metadata

### Subgroup performance decomposition
```python
tracker.blame(metric="accuracy", direction="decrease", slice_by="transaction_type")
# → "Version 3.2 caused a 12% accuracy drop specifically on 'online' transactions"
```

---

## Layer 3 — Visualization & Query *(planned)*

Makes the lineage graph explorable by humans.

### Visual DAG explorer
- Terminal-based: `mlineage dag <model>` renders an ASCII graph
- Web-based: lightweight local HTML viewer with d3-powered DAG

### Rich query API
```python
# Find all versions where accuracy dropped more than 2%
tracker.query(lambda v, prev: v.metric_delta(prev, "accuracy") < -0.02)

# Find all versions trained on data from a specific date range
tracker.query(lambda v, _: "2024-03" in v.dataset.path)

# Find versions with high embedding drift but stable metrics (silent semantic shift)
tracker.query(lambda v, prev: v.embedding_drift(prev) > 0.3 and abs(v.metric_delta(prev, "accuracy")) < 0.01)
```

### Connectors
These allow MLineage to read/write from existing MLOps infrastructure:
- **MLflow connector** — sync versions from MLflow Model Registry
- **HuggingFace Hub connector** — parse `base_model` lineage from model cards, write lineage back as metadata
- **DVC connector** — link `DataSnapshot` hashes to DVC-tracked datasets
- **W&B connector** — import run history as MLineage versions

---

## Layer 4 — Reproducibility *(future)*

The hardest layer, and the most valuable.

Given any node in the lineage graph, reconstruct the exact state that produced it:

```python
tracker.checkout("fraud-model-v3.2")
# → downloads checkpoint
# → fetches exact data snapshot
# → prints environment spec (Python version, dependencies, CUDA)
# → optionally: spins up a Docker container with the reconstructed environment
```

This requires:
- Storage backend with content-addressed artifact storage
- Environment snapshot capture (pip freeze, conda env, system info) at training time
- Integration with DVC or similar for data retrieval

---

## Community Priorities

The order of Layer 2 items is flexible. If you're working on a use case where one of these matters most, open an issue or discussion — community need shapes priority.

Current open questions where community input is most valuable:

1. **Storage backend format** — SQLite vs JSON vs something else? What fits best into existing ML project structures?
2. **Probe set design** — what's the right API for defining and managing behavioral fingerprints?
3. **Embedding drift** — which similarity metrics are most useful in practice? CKA is theoretically sound but computationally heavy.
4. **Connector priority** — MLflow or HuggingFace Hub first?

See [CONTRIBUTING.md](CONTRIBUTING.md) to weigh in.
