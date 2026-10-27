"""Static publication artifacts from completed aggregate files."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.controlled_geometry.common import read_json

COLORS = ["#2563a6", "#d35f40", "#c5ced8"]
POLICIES = ("early_exit", "always_rectify", "agreement")
NAMES = ("Early exit", "Always rectify", "Agreement")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert (root / "completion.json").exists()
    out = Path(__file__).resolve().parents[2] / "docs/experiments/2026-10-27-mobilenet-followup"
    summary = read_json(root / "report/summary.json")
    accuracy, timing = summary["accuracy"], summary["timing"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
    for row, dataset in enumerate(("validation", "benchmark")):
        for col, mode in enumerate(("legacy", "clahe")):
            ax = axes[row, col]
            paths = accuracy[f"crop-{dataset}-{mode}"]["paths"]
            bottom = np.zeros(3)
            for metric, label, color in zip(
                ("EDR", "FDR", "NRR"), ("Correct", "Wrong", "No return"), COLORS, strict=True
            ):
                values = np.array([paths[p][metric] for p in POLICIES])
                ax.bar(NAMES, values, bottom=bottom, color=color, label=label, width=0.6)
                if metric == "EDR":
                    for i, value in enumerate(values):
                        ax.text(i, value / 2, f"{value:.2f}%", ha="center", color="white")
                if metric == "FDR":
                    for i, value in enumerate(values):
                        ax.text(i, 103, f"Wrong: {value:.2f}%", ha="center", color=COLORS[1])
                bottom += values
            ax.set_title(f"{dataset.title()} / decoder CLAHE {'on' if mode == 'clahe' else 'off'}")
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            if col == 0:
                ax.set_ylabel("All inputs (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("MobileNet M, seed 42: crop policy tradeoffs", fontsize=15)
    fig.savefig(out / "crop_policy_tradeoffs.png", dpi=180)
    fig.savefig(out / "crop_policy_tradeoffs.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, mode in zip(axes, ("legacy", "clahe"), strict=True):
        for policy, marker, label in zip(POLICIES, ("o", "s", "^"), NAMES, strict=True):
            xs, ys = [], []
            for size in (1280, 1600, 1920):
                key = f"scene-mobilenetv3-small-{size}-{mode}"
                x = timing[key + "-timing"][policy]["p95_ms"]
                y = accuracy[key]["paths"][policy]["EDR"]
                xs.append(x)
                ys.append(y)
                ax.annotate(
                    str(size), (x, y), xytext=(3, 5), textcoords="offset points", fontsize=8
                )
            ax.plot(xs, ys, marker=marker, label=label, linewidth=1.2)
        key = f"scene-resnet18-c3-1280-{mode}"
        ax.scatter(
            timing[key + "-timing"]["early_exit"]["p95_ms"],
            accuracy[key]["paths"]["early_exit"]["EDR"],
            marker="X",
            color="black",
            s=75,
            label="ResNet 1280 early exit",
        )
        ax.set_title(f"Decoder CLAHE {'on' if mode == 'clahe' else 'off'}")
        ax.set_xlabel("Full policy execution p95 (ms)")
        ax.set_ylabel("Correct / all 100 synthetic scenes (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Scene benchmark v2: accuracy and latency (development data)", fontsize=14)
    fig.savefig(out / "scene_accuracy_latency.png", dpi=180)
    fig.savefig(out / "scene_accuracy_latency.svg")
    plt.close(fig)

    inputs = read_json(root / "inputs.json")
    prior = list(csv.DictReader((Path(inputs["prior_campaign"]) / "report/results.csv").open()))
    names, means, stds = [], [], []
    for model in ("resnet18-c3", "mobilenetv3-small"):
        for preset in ("R", "M"):
            values = [
                float(r["EDR"])
                for r in prior
                if r["dataset"] == "benchmark" and r["model"] == model and r["preset"] == preset
            ]
            assert len(values) == 3
            names.append(("ResNet" if model.startswith("resnet") else "MobileNet") + " " + preset)
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.bar(names, means, yerr=stds, capsize=6, color=[COLORS[0]] * 2 + ["#2c8561"] * 2)
    for i, value in enumerate(means):
        ax.text(i, value - 8, f"{value:.2f}%", ha="center", color="white")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct / all 1,500 crops (%)")
    ax.set_title("Prior four-way comparison: mean ± sample SD over 3 training seeds")
    fig.savefig(out / "prior_four_way.png", dpi=180)
    fig.savefig(out / "prior_four_way.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
