from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_mi_bar(
    feature_names: list[str],
    mi_scores: np.ndarray,
    top_n: int = 20,
    out_path: str | None = None,
) -> None:
    """
    Bar plot of MI scores for top-n features.

    Parameters
    ----------
    feature_names:
        Names of all features.
    mi_scores:
        MI scores aligned with feature_names.
    top_n:
        Number of top features to display.
    out_path:
        Optional output path for saving the figure.
    """
    idx = np.argsort(mi_scores)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_scores = mi_scores[idx]

    plt.figure(figsize=(8, 0.4 * top_n + 2))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_scores)
    plt.yticks(y_pos, top_feats)
    plt.gca().invert_yaxis()
    plt.xlabel("Mutual information")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)

