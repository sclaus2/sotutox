# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dependency-light level-set redistancing helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def graph_signed_distance(
    points: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    *,
    zero_tol: float = 1.0e-14,
) -> np.ndarray:
    """Approximate signed distance by graph marching from P1 zero crossings.

    The front is initialized at all exact zero vertices and at linearly
    interpolated zero crossings on sign-changing mesh edges. Distances are then
    propagated with Dijkstra's algorithm over the mesh edge graph (SciPy's, from
    a virtual source joined to every seeded vertex by its start distance).
    """

    points = np.asarray(points, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (num_points, geometric_dim).")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (num_edges, 2).")
    if values.shape != (points.shape[0],):
        raise ValueError("values must have one scalar per point.")

    n = points.shape[0]
    distances = np.full(n, np.inf, dtype=np.float64)
    i, j = edges[:, 0], edges[:, 1]
    length = np.linalg.norm(points[i] - points[j], axis=1)
    keep = length > 0.0
    i, j, length = i[keep], j[keep], length[keep]

    vi, vj = values[i], values[j]
    abs_i, abs_j = np.abs(vi), np.abs(vj)
    distances[i[abs_i <= zero_tol]] = 0.0
    distances[j[abs_j <= zero_tol]] = 0.0
    denom = abs_i + abs_j
    crossing = (vi * vj < 0.0) & (denom > zero_tol)
    np.minimum.at(distances, i[crossing], length[crossing] * abs_i[crossing] / denom[crossing])
    np.minimum.at(distances, j[crossing], length[crossing] * abs_j[crossing] / denom[crossing])

    # Dijkstra over the edge graph from a virtual source (vertex n) joined to every seeded vertex by its start distance
    seeds = np.flatnonzero(np.isfinite(distances))
    if len(seeds) > 0:
        rows = np.concatenate([i, j, np.full(len(seeds), n, dtype=np.int64)])
        cols = np.concatenate([j, i, seeds])
        weights = np.concatenate([length, length, distances[seeds]])
        graph = coo_matrix((weights, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
        distances = dijkstra(graph, directed=True, indices=n)[:n]

    fallback = np.abs(values)
    distances = np.where(np.isfinite(distances), distances, fallback)
    signs = np.where(values < 0.0, -1.0, 1.0)
    signed = signs * distances
    signed[np.abs(values) <= zero_tol] = 0.0
    return signed
