import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures" / "v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_history():
    history_path = ROOT / "outputs" / "logs" / "v2" / "training_history_v2.json"
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["validation_loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Training loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, label="Validation loss")
    ax.set_title("SwiftMask V2 training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_history.png", dpi=200)
    plt.close(fig)


def plot_threshold_sweep():
    threshold_path = ROOT / "outputs" / "logs" / "threshold_optimization" / "threshold_comparison.csv"
    import csv

    rows = []
    with threshold_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    thresholds = [float(r["threshold"]) for r in rows]
    ces = [float(r["ces"]) for r in rows]
    quality = [float(r["quality_q"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, ces, marker="o", linewidth=2.2, color="tab:blue", label="CES")
    ax.plot(thresholds, quality, marker="s", linewidth=2.2, color="tab:orange", label="Quality Q")
    ax.set_title("Threshold sweep on validation set")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_sweep.png", dpi=200)
    plt.close(fig)


def plot_captured_summary():
    captured_path = ROOT / "outputs" / "captured_evaluation" / "captured_results.json"
    with captured_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    labels = ["mIoU", "Boundary F1", "Quality Q", "CES"]
    values = [result["miou"], result["boundary_f1"], result["quality_q"], result["ces"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:green", "tab:orange", "tab:red"])
    ax.set_title("Captured-photo evaluation summary")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "captured_summary.png", dpi=200)
    plt.close(fig)


def main():
    plot_training_history()
    plot_threshold_sweep()
    plot_captured_summary()
    print(f"Saved summary figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
