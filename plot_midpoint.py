"""Generate the three required midpoint comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
BACKENDS = ["faiss", "redis", "postgres"]
FILTERS = ["unfiltered", "50pct", "10pct", "1pct"]
FILTER_LABELS = ["Unfiltered", "50%", "10%", "1%"]
COLORS = {"faiss": "#1f77b4", "redis": "#d62728", "postgres": "#2ca02c"}
RAM_COLUMNS = {
    "faiss": "faiss_ram_mb",
    "redis": "redis_ram_mb",
    "postgres": "postgres_ram_mb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="100k")
    return parser.parse_args()


def load_results(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    builds = []
    missing = []
    for backend in BACKENDS:
        summary_path = RESULTS_DIR / f"{backend}_midpoint_summary_{suffix}.csv"
        build_path = RESULTS_DIR / f"{backend}_index_build_{suffix}.csv"
        if not summary_path.exists():
            missing.append(summary_path.name)
        else:
            summaries.append(pd.read_csv(summary_path))
        if not build_path.exists():
            missing.append(build_path.name)
        else:
            builds.append(pd.read_csv(build_path))
    if missing:
        raise FileNotFoundError("Missing midpoint results: " + ", ".join(missing))
    return pd.concat(summaries, ignore_index=True), pd.concat(builds, ignore_index=True)


def ordered(summary: pd.DataFrame, backend: str) -> pd.DataFrame:
    return summary[summary["backend"] == backend].set_index("filter").reindex(FILTERS)


def plot_latency_qps(summary: pd.DataFrame, suffix: str) -> None:
    x = np.arange(len(FILTERS))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for index, backend in enumerate(BACKENDS):
        frame = ordered(summary, backend)
        offset = (index - 1) * width
        axes[0].bar(x + offset, frame["p95_ms"], width, label=backend.upper(), color=COLORS[backend])
        axes[1].bar(x + offset, frame["qps"], width, label=backend.upper(), color=COLORS[backend])
    axes[0].set_title("p95 query latency")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("Query throughput")
    axes[1].set_ylabel("Queries per second")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(FILTER_LABELS)
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.legend()
    fig.suptitle(f"Midpoint latency and throughput ({suffix})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"midpoint_latency_qps_{suffix}.png", dpi=220)
    plt.close(fig)


def plot_recall(summary: pd.DataFrame, suffix: str) -> None:
    x = np.arange(len(FILTERS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for index, backend in enumerate(BACKENDS):
        frame = ordered(summary, backend)
        axis.bar(
            x + (index - 1) * width,
            frame["recall_at_10"],
            width,
            label=backend.upper(),
            color=COLORS[backend],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(FILTER_LABELS)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Recall@10")
    axis.set_title(f"Retrieval accuracy by filter selectivity ({suffix})")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"midpoint_recall_{suffix}.png", dpi=220)
    plt.close(fig)


def plot_resources(summary: pd.DataFrame, builds: pd.DataFrame, suffix: str) -> None:
    internal_ram = []
    docker_ram = []
    cpu = []
    build_time = []
    for backend in BACKENDS:
        frame = summary[summary["backend"] == backend]
        build = builds[builds["backend"] == backend].iloc[0]
        internal_ram.append(float(frame[RAM_COLUMNS[backend]].median()))
        docker_ram.append(float(frame["docker_ram_mb"].median()))
        cpu.append(float(frame["docker_cpu_pct"].mean()))
        build_time.append(float(build["index_build_s"]))

    x = np.arange(len(BACKENDS))
    labels = [backend.upper() for backend in BACKENDS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x - 0.18, internal_ram, 0.36, label="Engine/process RAM")
    axes[0].bar(x + 0.18, docker_ram, 0.36, label="Docker RAM")
    axes[0].set_title("Memory footprint")
    axes[0].set_ylabel("MB")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, cpu, 0.55, color=[COLORS[b] for b in BACKENDS])
    axes[1].set_title("Mean query CPU utilization")
    axes[1].set_ylabel("Percent")
    axes[2].bar(x, build_time, 0.55, color=[COLORS[b] for b in BACKENDS])
    axes[2].set_title("Index build time")
    axes[2].set_ylabel("Seconds")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle(f"Midpoint resource comparison ({suffix})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"midpoint_resources_{suffix}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, builds = load_results(args.suffix)
    plot_latency_qps(summary, args.suffix)
    plot_recall(summary, args.suffix)
    plot_resources(summary, builds, args.suffix)
    print(f"Saved midpoint comparison figures to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
