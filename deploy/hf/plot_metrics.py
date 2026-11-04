"""Plots a training run's metrics and summarises them.

    python3 deploy/hf/plot_metrics.py runs/pq-large/metrics.jsonl plots/pq-large

Writes `<prefix>.png`, a figure with three panels, and `<prefix>.json`, a short
summary for model cards:

* loss against optimizer step: the training loss (averaged over a moving
  window, because single batches are noisy) and the held-out validation loss,
  both in nats per code, with the loss of a uniform guess for reference;
* the latest validation loss of every codebook, grouped by residual level,
  which shows how much harder each finer level is to predict;
* throughput in positions per second, which shows whether the GPU stayed busy.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Progress reports averaged into one point of the training curve.
SMOOTHING_WINDOW = 8

# Codes per codebook; a uniform guess costs ln(CODEBOOK_SIZE) nats.
CODEBOOK_SIZE = 1024

# Band groups per residual level in the band-group tokenizer; with fewer
# codebooks than this, every codebook is its own level.
BAND_GROUPS = 4

COLOURS = {"train": "#c8692f", "validation": "#3b4ba8", "uniform": "#9a9aa6"}


def read_events(path):
    """Every event of a metrics log, skipping lines that are not JSON."""
    events = []
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def moving_average(values, window):
    """The mean of each value and the `window - 1` before it."""
    averaged = []
    for index in range(len(values)):
        recent = values[max(0, index - window + 1): index + 1]
        averaged.append(sum(recent) / len(recent))
    return averaged


def plot_losses(axis, progress, validation):
    """Training and validation loss against step."""
    steps = [event["step"] for event in progress]
    axis.plot(steps, moving_average([event["loss"] for event in progress], SMOOTHING_WINDOW),
              color=COLOURS["train"], linewidth=1.4, label="training (moving average)")
    axis.plot([event["step"] for event in validation], [event["loss"] for event in validation],
              color=COLOURS["validation"], marker="o", markersize=3, linewidth=1.4, label="validation")

    axis.axhline(math.log(CODEBOOK_SIZE), color=COLOURS["uniform"], linestyle="--", linewidth=1, label="uniform guess")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("loss (nats per code)")
    axis.set_title("Loss")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)


def plot_codebooks(axis, latest):
    """The latest validation loss of every codebook, coloured by residual level."""
    losses = latest["loss_per_codebook"]
    groups = BAND_GROUPS if len(losses) % BAND_GROUPS == 0 and len(losses) > BAND_GROUPS else 1
    levels = len(losses) // groups
    palette = plt.cm.copper_r([0.25 + 0.6 * level / max(levels - 1, 1) for level in range(levels)])

    colours = [palette[index // groups] for index in range(len(losses))]
    axis.bar(range(len(losses)), losses, color=colours)
    axis.axhline(math.log(CODEBOOK_SIZE), color=COLOURS["uniform"], linestyle="--", linewidth=1)

    axis.set_xlabel(f"codebook ({levels} levels × {groups} band groups, coarse to fine)" if groups > 1 else "codebook (coarse to fine)")
    axis.set_ylabel("validation loss (nats per code)")
    axis.set_title(f"Per codebook, step {latest['step']}")
    axis.grid(axis="y", alpha=0.25)


def plot_throughput(axis, progress):
    """Positions per second over the run."""
    axis.plot([event["step"] for event in progress], [event["positions_per_second"] for event in progress],
              color=COLOURS["train"], linewidth=1)

    axis.set_xlabel("optimizer step")
    axis.set_ylabel("positions per second")
    axis.set_title("Throughput")
    axis.grid(alpha=0.25)


def summary(name, events, progress, validation):
    """The numbers a model card needs."""
    started = [event for event in events if event["event"] == "started"]
    best = min(validation, key=lambda event: event["loss"]) if validation else None

    return {
        "run": name,
        "parameters": started[-1]["parameters"] if started else None,
        "train_tracks": started[-1]["train_tracks"] if started else None,
        "latest_step": progress[-1]["step"] if progress else 0,
        "latest_train_loss": round(progress[-1]["loss"], 4) if progress else None,
        "best_validation_loss": round(best["loss"], 4) if best else None,
        "best_validation_step": best["step"] if best else None,
        "milliseconds_per_step": round(progress[-1]["milliseconds_per_step"], 1) if progress else None,
    }


def main(metrics, prefix):
    events = read_events(metrics)
    progress = [event for event in events if event["event"] == "progress"]
    validation = [event for event in events if event["event"] == "validation"]
    name = Path(prefix).name

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plot_losses(axes[0], progress, validation)
    if validation:
        plot_codebooks(axes[1], validation[-1])
    plot_throughput(axes[2], progress)

    figure.suptitle(f"StaleyNET run {name}", fontsize=12)
    figure.tight_layout()
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(f"{prefix}.png", dpi=130)

    Path(f"{prefix}.json").write_text(json.dumps(summary(name, events, progress, validation), indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
