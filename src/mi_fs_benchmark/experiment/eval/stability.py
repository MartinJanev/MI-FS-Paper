from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def jaccard_similarity(a: Iterable[int], b: Iterable[int]) -> float:
    """Compute Jaccard similarity between two sets of feature indices."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def mean_pairwise_jaccard(feature_sets: Sequence[Iterable[int]]) -> float:
    """
    Mean pairwise Jaccard similarity over a list of selected feature sets.

    Parameters
    ----------
    feature_sets:
        Sequence of index collections, one per fold.

    Returns
    -------
    float
    """
    if len(feature_sets) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(feature_sets)):
        for j in range(i + 1, len(feature_sets)):
            sims.append(jaccard_similarity(feature_sets[i], feature_sets[j]))
    return float(np.mean(sims))
