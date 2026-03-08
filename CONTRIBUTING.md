# Contributing to MLineage

First of all: thank you for being here. MLineage is being built in public from the very beginning, and every form of contribution matters — not just code.

---

## Ways to Contribute

### 🗣️ Share Your Pain (Most Valuable Right Now)

The most useful thing you can do at this stage is **tell us about your experience** tracking continual learning models.

- Open a [Discussion](https://github.com/Fredbcx/mlineage/discussions) and describe how you currently track model evolution
- Tell us what your biggest frustration is with existing tools
- Share a workflow or workaround you've built — we want to understand the problem space deeply before locking in API design

This kind of input directly shapes the library's direction.

### 🐛 Open Issues

Found a bug? Have a feature request? Open an issue. Use the templates provided and be as specific as possible.

For feature requests, the most useful information is: *what problem are you trying to solve?* Not just *what feature do you want?*

### 📖 Improve Documentation

Docs are often the first thing contributors can improve without deep codebase knowledge:
- Fix typos or unclear explanations
- Add examples to the docstrings
- Write a tutorial for a specific use case

### 🔧 Submit Code

See the development setup below. Before writing code for a new feature, open an issue first to discuss the approach. This avoids wasted effort on PRs that go in a different direction than the project.

Good first issues are labeled [`good first issue`](https://github.com/Fredbcx/mlineage/issues?q=is%3Aissue+label%3A%22good+first+issue%22).

---

## Development Setup

### Prerequisites

- Python 3.9+
- [mypy](https://mypy.readthedocs.io/) for type checking (required — all code must pass strict mypy)
- [pytest](https://docs.pytest.org/) for tests

### Installation

```bash
git clone https://github.com/Fredbcx/mlineage.git
cd mlineage
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy mlineage/ --strict
```

All PRs must pass both `pytest` and `mypy --strict` before they can be merged.

### Code Style

- We use [black](https://github.com/psf/black) for formatting and [ruff](https://github.com/astral-sh/ruff) for linting
- Run `black mlineage/` and `ruff check mlineage/` before committing
- Pre-commit hooks are available: `pre-commit install`

---

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Name your branch descriptively: `feature/mlflow-connector`, `fix/snapshot-hash-collision`
3. Write tests for any new functionality
4. Ensure `pytest`, `mypy --strict`, `black`, and `ruff` all pass
5. Open a PR with a clear description of what it does and why
6. Link any related issues

PRs will be reviewed as time allows — this is a side project, so turnaround may not be instant. We'll aim for a response within a week.

---

## Project Principles

When contributing, please keep these in mind:

- **Non-intrusive first** — features should integrate into existing workflows, not replace them
- **Temporal-first** — time and ordering are core, not optional
- **Composability over completeness** — a focused connector that does one thing well is better than a sprawling integration
- **Type safety** — all public APIs must be fully typed and pass `mypy --strict`

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Be kind, be constructive, assume good intent.

---

## Questions?

Open a [Discussion](https://github.com/Fredbcx/mlineage/discussions) — don't be shy.
