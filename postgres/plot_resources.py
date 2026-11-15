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
    summary_df = load_summary_data(DATASET_SUFFIX)
    build_df = load_build_data(DATASET_SUFFIX)
    summary_df = summary_df[summary_df["backend"] == "postgres"]
    build_df = build_df[build_df["backend"] == "postgres"]
    if summary_df.empty or build_df.empty:
        raise ValueError(
            "PostgreSQL summary and index-build CSVs are required. "
            "Run postgres/bench_selectivity.py and postgres/load_and_index.py first."
        )

    short_labels = ["Unfiltered", "50% Filter", "10% Filter", "1% Filter"]
    x = np.arange(len(FILTER_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    summary = summary_df.set_index("filter").reindex(FILTER_ORDER)
    width = 0.35
    internal_ram = axes[0].bar(
        x - width / 2,
        summary["postgres_ram_mb"],
        width=width,
        label="Internal Used RAM (MB)",
        color="#1f77b4",
    )
    docker_ram = axes[0].bar(
        x + width / 2,
        summary["docker_ram_mb"],
        width=width,
        label="Docker Container RAM (MB)",
        color="#6baed6",
    )
    axes[0].bar_label(internal_ram, fmt="%.1f", padding=2, fontsize=8)
    axes[0].bar_label(docker_ram, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels)
    axes[0].set_ylabel("Memory (MB)")
    axes[0].set_title(f"Memory Footprint ({DATASET_SUFFIX})", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    cpu_bars = axes[1].bar(x, summary["docker_cpu_pct"], width=0.45, color="#ff7f0e")
    axes[1].bar_label(cpu_bars, fmt="%.2f%%", padding=3, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels)
    axes[1].set_ylabel("CPU Utilization (%)")
    axes[1].set_title(f"Docker CPU Utilization ({DATASET_SUFFIX})", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    build_x = np.arange(len(build_df))
    colors = [BACKEND_COLORS.get(backend, "#9467bd") for backend in build_df["backend"]]
    build_bars = axes[2].bar(
        build_x, build_df["index_build_s"], width=0.4, color=colors
    )
    axes[2].bar_label(build_bars, fmt="%.2f s", padding=3, fontsize=9)
    axes[2].set_xticks(build_x)
    axes[2].set_xticklabels([backend.upper() for backend in build_df["backend"]])
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
