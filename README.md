# MLineage 🧬

> **Track how your models evolve, not just what they are.**

[![Status](https://img.shields.io/badge/status-early%20development-orange)](https://github.com/Fredbcx/mlineage)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)

---

## The Problem

Have you ever looked at a model in production and asked:

- *"Why is this model behaving differently than last month?"*
- *"Which training batch introduced this performance regression?"*
- *"What data was this model trained on, exactly, and in what order?"*
- *"The metrics look stable — so why are users complaining about worse recommendations?"*
- *"If I need to reproduce the model from 3 weeks ago, where do I even start?"*

If you're doing **continual learning** — models that update on new data over time — existing MLOps tools leave you in the dark.

MLflow, W&B, and DVC are great at tracking experiments. They treat every model version as a static artifact. But a model that learns continuously is not a static artifact. It's a living history of data and decisions.

**MLineage fills that gap.**

---

## What MLineage Does

MLineage tracks the **temporal evolution** of your models as a first-class concept:

```
Model v1.0  ──[trained on dataset-jan]──▶  Model v1.1
    │                                           │
    │                                    [updated on dataset-feb]
    │                                           │
    └──[fine-tuned]──▶  Model v2.0 ◀────  Model v1.2
                                    [updated on dataset-mar]
```

For each node in this graph, MLineage records:

- **Model version** — hash, path, framework, architecture metadata
- **Data snapshot** — which data was used, in what order, with what hash
- **Training context** — hyperparameters, environment, duration
- **Performance delta** — how metrics changed relative to the previous version
- **Semantic fingerprint** *(planned)* — how the model's internal representations shifted
- **Causal annotations** — why this update happened (human notes or automated tags)

Then it lets you **query this history**:

```python
import mlineage as ml

tracker = ml.Tracker.load("my-fraud-model")

# Walk the full ancestry of a model
for version in tracker.lineage("fraud-model-v3.2").ancestors():
    print(version)

# git-blame for models: which update caused the accuracy drop?
culprit = tracker.blame(metric="accuracy", direction="decrease")
print(f"Regression introduced in: {culprit.notes}")

# Reproduce the exact model state from 6 weeks ago  [planned]
tracker.checkout("fraud-model-v3.2", at="2024-01-15")
```

---

## Debugging Model Regressions

One of MLineage's core use cases is acting as **`git blame` for ML models**.

When a model misbehaves in production, the standard debugging workflow is painful: you scan through experiment logs, compare checkpoints manually, and try to correlate metric drops with data changes. With MLineage, the question *"which update broke this?"* has a direct answer:

```python
# Find the version that caused the largest accuracy drop
culprit = tracker.blame(metric="accuracy", direction="decrease")

# Inspect what changed at that version
print(culprit.notes)           # human annotation from the engineer who ran it
print(culprit.dataset)         # exact data snapshot used
print(culprit.hyperparameters) # training configuration

# Compare metrics before and after
history = tracker.history()
for i, v in enumerate(history[1:], 1):
    delta = v.metric_delta(history[i-1], "accuracy")
    print(f"{v.id[:8]}: Δaccuracy = {delta:+.4f}")
```

This doesn't replace proper monitoring — but it dramatically shortens the time from *"something is wrong"* to *"here is what changed and why"*.

---

## Beyond Metrics: Semantic Shift

*(This section describes planned functionality — not yet implemented.)*

Aggregate metrics can be silent about the most dangerous kind of model change: **semantic drift**.

A fraud detection model might maintain 94% accuracy across updates while quietly shifting its internal representation of what "suspicious" means — becoming more sensitive to one fraud pattern and blind to another. The accuracy number stays flat. The model has fundamentally changed.

MLineage's planned analysis layer will surface this through three lenses:

**Behavioral fingerprinting** — track how the model responds to a fixed set of "probe" inputs across versions. If predictions on edge cases shift, something has changed even if aggregate metrics haven't.

**Embedding drift** — for models with internal representations (transformers, neural networks), measure how much the representation space has rotated or translated between versions. A large embedding drift with stable metrics is a red flag.

**Subgroup performance tracking** — decompose metric changes by input subgroups. Aggregate accuracy can hide a model that got better on the majority class and worse on a rare but critical minority.

The goal is to make it possible to say: *"Version 3.2 shifted its understanding of fraud — here is the evidence."*

---

## Statistical Analysis Layer

*(Planned — not yet implemented.)*

MLineage will provide statistical tools to characterize *both* data and model changes across versions, making the lineage graph queryable not just structurally but analytically.

**For data snapshots:**
- Distribution shift between training batches (PSI, KL divergence, Kolmogorov-Smirnov per feature)
- Automatic flagging of batches with anomalous statistical profiles
- Label distribution tracking over time

**For model versions:**
- Confidence calibration drift — is the model becoming over- or under-confident?
- Performance decomposition by slice, cohort, or time window
- Correlation between data distribution shift and metric delta

The design principle here is **wrapping, not reimplementing**: tools like [Evidently](https://github.com/evidentlyai/evidently) and [alibi-detect](https://github.com/SeldonIO/alibi-detect) already do drift detection well. MLineage's job is to connect their output to the lineage graph, so that a statistical anomaly is immediately traceable to a specific data snapshot and model version.

---

## Who Is This For

MLineage is designed for teams doing **production continual learning**:

- 🎯 Recommendation systems that retrain on user feedback
- 🔍 Fraud detection models that adapt to new attack patterns
- 🤖 LLMs fine-tuned in feedback loops
- 📈 Any model where "version" is a point in a continuous stream, not a discrete release

If you're doing one-shot training, existing tools probably cover you fine. If you're updating models continuously, you likely have a folder of checkpoints and a spreadsheet. MLineage is the upgrade.

---

## Current Status

> ⚠️ **This project is in early development.** The API is not stable. Nothing is production-ready. But the design is taking shape and contributions are very welcome.

What exists today:
- [x] Core data model (`ModelVersion`, `DataSnapshot`, `LineageGraph`)
- [x] Basic tracker API with context-manager interface
- [x] `blame()` — identify the version that caused a metric regression
- [x] Ancestor/descendant traversal on the lineage DAG
- [x] 20 unit tests, all passing

What's planned:
- [ ] Local storage backend (persistence across sessions)
- [ ] MLflow connector
- [ ] HuggingFace Hub connector
- [ ] DVC connector
- [ ] S3/GCS storage backends
- [ ] CLI for querying lineage
- [ ] Distribution shift analysis (via Evidently)
- [ ] Behavioral fingerprinting
- [ ] Embedding drift detection
- [ ] Visual DAG explorer

See [ROADMAP.md](ROADMAP.md) for details and priorities.

---

## Installation

```bash
# Not yet on PyPI — install from source
git clone https://github.com/Fredbcx/mlineage.git
cd mlineage
pip install -e ".[dev]"

# Optional integrations
pip install -e ".[mlflow]"       # MLflow connector
pip install -e ".[huggingface]"  # HuggingFace Hub connector
pip install -e ".[analysis]"     # Drift detection (Evidently)
```

---

## Quick Start

```python
import mlineage as ml

# Initialize a tracker for a model
tracker = ml.Tracker("fraud-detector", storage="./lineage-store")

# Log a model version after training
with tracker.log_version() as v:
    v.model_path = "./checkpoints/model_20240201.pt"
    v.dataset = ml.DataSnapshot(path="./data/jan_2024/", hash="sha256:abc123")
    v.metrics = {"accuracy": 0.94, "f1": 0.91}
    v.notes = "Monthly update with January fraud patterns"

# Log next month's update — parent is set automatically
with tracker.log_version() as v:
    v.model_path = "./checkpoints/model_20240301.pt"
    v.dataset = ml.DataSnapshot(path="./data/feb_2024/", hash="sha256:def456")
    v.metrics = {"accuracy": 0.91, "f1": 0.88}
    v.notes = "March update — possible data quality issue"

# Investigate the regression
culprit = tracker.blame(metric="accuracy", direction="decrease")
print(f"Regression in: {culprit.notes}")
print(f"Dataset used:  {culprit.dataset}")
```

---

## Design Principles

1. **Non-intrusive** — drop into existing training pipelines with minimal changes
2. **Composable** — works alongside MLflow, W&B, DVC, not instead of them
3. **Temporal-first** — time and order are primary, not an afterthought
4. **Reproducible** — every version can be reconstructed from its lineage record
5. **Queryable** — lineage data is only useful if you can ask questions of it
6. **Wrap, don't reimplement** — integrate best-in-class libraries for drift detection and analysis rather than building from scratch

---

## Contributing

This project is being built in public from day one. If you've felt the pain of tracking continual learning models — or if you've hit the limits of existing tools when doing semantic analysis of model drift — your experience is directly useful here.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved, including ways to contribute that don't require writing code.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
