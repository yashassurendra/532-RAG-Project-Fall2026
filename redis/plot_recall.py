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
    w = 0.45 if len(backends) == 1 else 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, b in enumerate(backends):
        sub = df[df["backend"] == b].set_index("filter").reindex(FILTER_ORDER)
        offset = 0 if len(backends) == 1 else (i - (len(backends) - 1) / 2) * w
        bars = ax.bar(
            x + offset,
            sub["recall_at_10"],
            width=w,
            label=b.upper(),
            color=BACKEND_COLORS.get(b, "#2ca02c"),
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(FILTER_LABELS)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Exact Ground Truth (1.0)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Retrieval Accuracy (Recall@10) by Filter Selectivity ({DATASET_SUFFIX})", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"midpoint_recall_{DATASET_SUFFIX}.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved Recall@10 plot -> {out_path}")


if __name__ == "__main__":
    main()