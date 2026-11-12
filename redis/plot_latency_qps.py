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

    # Left Subplot: Latency (p50, p95, p99 for single backend; p95 comparison for multi-backend)
    if len(backends) == 1:
        b = backends[0]
        sub = df[df["backend"] == b].set_index("filter").reindex(FILTER_ORDER)
        w = 0.25
        b1 = axes[0].bar(x - w, sub["p50_ms"], width=w, label="p50 (Median)", color="#2b5c8f")
        b2 = axes[0].bar(x, sub["p95_ms"], width=w, label="p95", color="#e68a00")
        b3 = axes[0].bar(x + w, sub["p99_ms"], width=w, label="p99 (Tail)", color="#c43c39")
        axes[0].bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
        axes[0].bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
        axes[0].bar_label(b3, fmt="%.2f", padding=2, fontsize=8)
        axes[0].set_title(f"{b.upper()} Query Latency by Selectivity ({DATASET_SUFFIX})", fontweight="bold")
    else:
        w = 0.8 / len(backends)
        for i, b in enumerate(backends):
            sub = df[df["backend"] == b].set_index("filter").reindex(FILTER_ORDER)
            offset = (i - (len(backends) - 1) / 2) * w
            bars = axes[0].bar(
                x + offset,
                sub["p95_ms"],
                width=w,
                label=f"{b.upper()} (p95)",
                color=BACKEND_COLORS.get(b, None),
            )
            axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        axes[0].set_title(f"p95 Query Latency by Backend ({DATASET_SUFFIX})", fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FILTER_LABELS, rotation=12)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    # Right Subplot: Throughput (QPS)
    w_qps = 0.45 if len(backends) == 1 else 0.8 / len(backends)
    for i, b in enumerate(backends):
        sub = df[df["backend"] == b].set_index("filter").reindex(FILTER_ORDER)
        offset = 0 if len(backends) == 1 else (i - (len(backends) - 1) / 2) * w_qps
        bars = axes[1].bar(
            x + offset,
            sub["qps"],
            width=w_qps,
            label=b.upper(),
            color=BACKEND_COLORS.get(b, "#2b5c8f"),
        )
        axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(FILTER_LABELS, rotation=12)
    axes[1].set_ylabel("Throughput (Queries / Sec)")
    axes[1].set_title(f"Query Throughput (QPS) by Selectivity ({DATASET_SUFFIX})", fontweight="bold")
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