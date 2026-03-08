"""
examples/fraud_detector_continual.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A realistic example of using MLineage to track a fraud detection model
that updates monthly on new data — a classic continual learning scenario.
"""

import mlineage as ml

def main() -> None:
    # Initialize a tracker for the fraud detection model.
    # All lineage data will be stored in ./lineage-store (local, for now).
    tracker = ml.Tracker("fraud-detector", storage="./lineage-store")

    print("=== Simulating 3 months of continual learning ===\n")

    # --- Month 1: Initial training ---
    with tracker.log_version() as v:
        v.model_path = "./checkpoints/fraud_jan.pt"
        v.dataset = ml.DataSnapshot(
            path="./data/2024-01/",
            hash="sha256:1a2b3c",
            record_count=150_000,
            notes="January transactions, post-holiday fraud spike"
        )
        v.metrics = {"accuracy": 0.921, "f1": 0.887, "false_positive_rate": 0.032}
        v.hyperparameters = {"lr": 1e-3, "epochs": 10, "batch_size": 512}
        v.notes = "Initial model trained on January data"
        v.tags = ["baseline", "jan-2024"]

    jan_version = v.version
    print(f"Logged: {jan_version}")

    # --- Month 2: Update on February data ---
    with tracker.log_version() as v:
        v.model_path = "./checkpoints/fraud_feb.pt"
        v.dataset = ml.DataSnapshot(
            path="./data/2024-02/",
            hash="sha256:4d5e6f",
            record_count=142_000,
            notes="February transactions, new card-skimming pattern detected"
        )
        v.metrics = {"accuracy": 0.934, "f1": 0.901, "false_positive_rate": 0.028}
        v.hyperparameters = {"lr": 5e-4, "epochs": 5, "batch_size": 512}
        v.notes = "February update: adapted to new card-skimming patterns"
        v.tags = ["feb-2024"]

    feb_version = v.version
    print(f"Logged: {feb_version}")

    # --- Month 3: Update causes regression ---
    with tracker.log_version() as v:
        v.model_path = "./checkpoints/fraud_mar.pt"
        v.dataset = ml.DataSnapshot(
            path="./data/2024-03/",
            hash="sha256:7g8h9i",
            record_count=160_000,
            notes="March transactions — data pipeline had a bug, some labels inverted"
        )
        v.metrics = {"accuracy": 0.891, "f1": 0.854, "false_positive_rate": 0.047}
        v.hyperparameters = {"lr": 5e-4, "epochs": 5, "batch_size": 512}
        v.notes = "March update (WARNING: label quality issues discovered post-hoc)"
        v.tags = ["mar-2024", "data-quality-issue"]

    mar_version = v.version
    print(f"Logged: {mar_version}")

    # --- Query the history ---
    print("\n=== Model History ===")
    print(tracker.summary())

    # --- Blame: which version caused the accuracy drop? ---
    print("\n=== Blame Analysis: accuracy regression ===")
    culprit = tracker.blame(metric="accuracy", direction="decrease")
    if culprit:
        print(f"Largest accuracy drop introduced in: {culprit.id[:8]}...")
        print(f"  Notes: {culprit.notes}")
        print(f"  Dataset: {culprit.dataset}")

    # --- View ancestors of the march model ---
    print("\n=== Ancestors of the March model ===")
    graph = tracker._graph
    for ancestor in graph.ancestors(mar_version.id):
        print(f"  {ancestor.id[:8]}... | {ancestor.notes}")

    print("\nDone.")


if __name__ == "__main__":
    main()
