import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    DATASET_SUFFIX,
    RESULTS_DIR,
    FILTER_ORDER,
    FILTER_LABELS,
    BACKEND_COLORS,
    load_summary_data,
)


def main():
    df = load_summary_data(DATASET_SUFFIX)
    backends = df["backend"].unique()
    x = np.arange(len(FILTER_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if len(backends) == 1:
        backend = backends[0]
        sub = df[df["backend"] == backend].set_index("filter").reindex(FILTER_ORDER)
        width = 0.25
        p50 = axes[0].bar(
            x - width, sub["p50_ms"], width=width, label="p50 (Median)", color="#2b5c8f"
        )
        p95 = axes[0].bar(x, sub["p95_ms"], width=width, label="p95", color="#e68a00")
        p99 = axes[0].bar(
            x + width, sub["p99_ms"], width=width, label="p99 (Tail)", color="#c43c39"
        )
        axes[0].bar_label(p50, fmt="%.2f", padding=2, fontsize=8)
        axes[0].bar_label(p95, fmt="%.2f", padding=2, fontsize=8)
        axes[0].bar_label(p99, fmt="%.2f", padding=2, fontsize=8)
        axes[0].set_title(
            f"{backend.upper()} Query Latency by Selectivity ({DATASET_SUFFIX})",
            fontweight="bold",
        )
    else:
        width = 0.8 / len(backends)
        for i, backend in enumerate(backends):
            sub = df[df["backend"] == backend].set_index("filter").reindex(FILTER_ORDER)
            offset = (i - (len(backends) - 1) / 2) * width
            bars = axes[0].bar(
                x + offset,
                sub["p95_ms"],
                width=width,
                label=f"{backend.upper()} (p95)",
                color=BACKEND_COLORS.get(backend),
            )
            axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axes[0].set_title(
            f"p95 Query Latency by Backend ({DATASET_SUFFIX})", fontweight="bold"
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FILTER_LABELS, rotation=12)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    qps_width = 0.45 if len(backends) == 1 else 0.8 / len(backends)
    for i, backend in enumerate(backends):
        sub = df[df["backend"] == backend].set_index("filter").reindex(FILTER_ORDER)
        offset = 0 if len(backends) == 1 else (i - (len(backends) - 1) / 2) * qps_width
        bars = axes[1].bar(
            x + offset,
            sub["qps"],
            width=qps_width,
            label=backend.upper(),
            color=BACKEND_COLORS.get(backend, "#2b5c8f"),
        )
        axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(FILTER_LABELS, rotation=12)
    axes[1].set_ylabel("Throughput (Queries / Sec)")
    axes[1].set_title(
        f"Query Throughput (QPS) by Selectivity ({DATASET_SUFFIX})", fontweight="bold"
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    if len(backends) > 1:
        axes[1].legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / f"midpoint_latency_qps_{DATASET_SUFFIX}.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved Latency & QPS plot -> {out_path}")


if __name__ == "__main__":
    main()
