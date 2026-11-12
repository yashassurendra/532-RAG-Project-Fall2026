import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    DATASET_SUFFIX,
    RESULTS_DIR,
    FILTER_ORDER,
    BACKEND_COLORS,
    load_summary_data,
    load_build_data,
)


def main():
    sum_df = load_summary_data(DATASET_SUFFIX)
    bld_df = load_build_data(DATASET_SUFFIX)

    short_labels = ["Unfiltered", "50% Filter", "10% Filter", "1% Filter"]
    x = np.arange(len(FILTER_ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Subplot 1: Memory Usage (Internal vs Docker Container RAM)
    primary_backend = sum_df["backend"].iloc[0]
    sub = sum_df[sum_df["backend"] == primary_backend].set_index("filter").reindex(FILTER_ORDER)
    w = 0.35
    b1 = axes[0].bar(x - w / 2, sub["redis_ram_mb"], width=w, label="Internal Used RAM (MB)", color="#1f77b4")
    b2 = axes[0].bar(x + w / 2, sub["docker_ram_mb"], width=w, label="Docker Container RAM (MB)", color="#6baed6")
    axes[0].bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    axes[0].bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels)
    axes[0].set_ylabel("Memory (MB)")
    axes[0].set_title(f"Memory Footprint ({DATASET_SUFFIX})", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    # Subplot 2: Docker CPU % During Querying
    b_cpu = axes[1].bar(x, sub["docker_cpu_pct"], width=0.45, color="#ff7f0e")
    axes[1].bar_label(b_cpu, fmt="%.2f%%", padding=3, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels)
    axes[1].set_ylabel("CPU Utilization (%)")
    axes[1].set_title(f"Docker CPU Utilization ({DATASET_SUFFIX})", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: HNSW Index Build Time
    bx = np.arange(len(bld_df))
    colors = [BACKEND_COLORS.get(b, "#9467bd") for b in bld_df["backend"]]
    b_bld = axes[2].bar(bx, bld_df["index_build_s"], width=0.4, color=colors)
    axes[2].bar_label(b_bld, fmt="%.2f s", padding=3, fontsize=9)
    axes[2].set_xticks(bx)
    axes[2].set_xticklabels([b.upper() for b in bld_df["backend"]])
    axes[2].set_ylabel("Build Time (seconds)")
    axes[2].set_title(f"Index Construction Time ({DATASET_SUFFIX})", fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"midpoint_resources_{DATASET_SUFFIX}.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved Resources & Build Time plot -> {out_path}")


if __name__ == "__main__":
    main()