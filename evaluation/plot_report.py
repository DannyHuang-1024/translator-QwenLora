from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create report figures from evaluation artifacts.")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--training-metrics", default=None)
    return parser.parse_args()


def read_training_metrics(path: Path):
    records = []
    if not path or not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    output_dir = Path(args.evaluation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    metrics = pd.read_csv(output_dir / "sentence_metrics.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = [0, 1]
    width = 0.36
    ax.bar([value - width / 2 for value in x], [summary["base_bleu_char"], summary["lora_bleu_char"]], width, label="BLEU")
    ax.bar([value + width / 2 for value in x], [summary["base_chrf"], summary["lora_chrf"]], width, label="chrF")
    ax.set_xticks(x, ["Base Qwen3", "LoRA fine-tuned"])
    ax.set_ylabel("Score")
    ax.set_title("Translation quality on the held-out test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_metrics.png", dpi=200)
    fig.savefig(output_dir / "model_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(metrics["chrf_delta"], bins=20, color="#2f6f9f", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("LoRA chrF - base chrF")
    ax.set_ylabel("Number of examples")
    ax.set_title("Per-example translation improvement")
    fig.tight_layout()
    fig.savefig(output_dir / "sentence_delta.png", dpi=200)
    fig.savefig(output_dir / "sentence_delta.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    counts = [summary["sentence_chrf_wins"], summary["sentence_chrf_ties"], summary["sentence_chrf_losses"]]
    bars = ax.bar(["LoRA better", "Tie", "Base better"], counts, color=["#3a7d44", "#8c8c8c", "#b44c4c"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of test examples")
    ax.set_title("Paired sentence-level comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "win_tie_loss.png", dpi=200)
    fig.savefig(output_dir / "win_tie_loss.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(metrics["reference_chars"], metrics["base_chars"], alpha=0.45, label="Base")
    ax.scatter(metrics["reference_chars"], metrics["lora_chars"], alpha=0.45, label="LoRA")
    max_length = max(metrics[["reference_chars", "base_chars", "lora_chars"]].max())
    ax.plot([0, max_length], [0, max_length], "k--", linewidth=1)
    ax.set_xlabel("Reference length (Chinese characters)")
    ax.set_ylabel("Generated length")
    ax.set_title("Output length comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "length_comparison.png", dpi=200)
    fig.savefig(output_dir / "length_comparison.pdf")
    plt.close(fig)

    bins = pd.cut(
        metrics["reference_chars"],
        bins=[-1, 20, 40, 80, float("inf")],
        labels=["0-20", "21-40", "41-80", "81+"],
    )
    grouped = metrics.assign(length_bucket=bins).groupby("length_bucket", observed=False)["chrf_delta"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(grouped.index.astype(str), grouped.values, color="#2f6f9f")
    ax.axhline(0, color="black", linewidth=1)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlabel("Reference length (Chinese characters)")
    ax.set_ylabel("Mean chrF improvement")
    ax.set_title("LoRA improvement by sentence length")
    fig.tight_layout()
    fig.savefig(output_dir / "length_bucket_delta.png", dpi=200)
    fig.savefig(output_dir / "length_bucket_delta.pdf")
    plt.close(fig)

    if args.training_metrics:
        records = read_training_metrics(Path(args.training_metrics))
        train = [r for r in records if "loss" in r]
        evaluation = [r for r in records if "eval_loss" in r]
        if train or evaluation:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            if train:
                ax.plot([r["step"] for r in train], [r["loss"] for r in train], label="Training loss")
            if evaluation:
                ax.plot([r["step"] for r in evaluation], [r["eval_loss"] for r in evaluation], marker="o", label="Evaluation loss")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Loss")
            ax.set_title("Training and evaluation loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "training_curves.png", dpi=200)
            fig.savefig(output_dir / "training_curves.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()
