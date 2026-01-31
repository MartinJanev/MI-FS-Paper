# src/mi_fs_benchmark/fs/mrmr.py
"""
This is a real greedy mRMR:

First compute relevance: I(X_j; Y) for all j (MI with target).

Compute pairwise redundancy: I(X_j; X_k) for all pairs using mutual_info_regression.

Greedily build a set S:

Start with feature having highest I(X_j; Y).

At each step, choose j maximizing
score(j) = I(X_j; Y) - λ * mean_{s∈S} I(X_j; X_s).

Note: This is O(d²) to build the MI matrix, so you might want to use it on reduced candidate sets for very high dimensional data.
"""

# Deprecated: implementation consolidated in core.fs.__init__
from .__init__ import MRMRSelector  # noqa: F401
