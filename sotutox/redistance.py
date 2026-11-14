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


def _segment_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances of the points p (n, d) to the segments (a, b) (n, d) each."""
    ab = b - a
    t = np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1.0e-300)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _triangle_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distances of the points p (n, 3) to the triangles (a, b, c) (n, 3) each: the plane's where the foot falls inside
    the triangle, else the nearest edge's."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    unit = n / np.maximum(nn, 1.0e-300)[:, None]
    h = np.einsum("ij,ij->i", p - a, unit)
    q = p - h[:, None] * unit
    inside = (
        (np.einsum("ij,ij->i", np.cross(b - a, q - a), unit) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(c - b, q - b), unit) >= 0.0)
        & (np.einsum("ij,ij->i", np.cross(a - c, q - c), unit) >= 0.0)
        & (nn > 0.0)
    )
    edges = np.minimum(np.minimum(_segment_distances(p, a, b), _segment_distances(p, b, c)), _segment_distances(p, c, a))
    return np.where(inside, np.abs(h), edges)


def exact_signed_distance(
    points: np.ndarray,
    facets: np.ndarray,
    signs: np.ndarray,
    *,
    comm=None,
    candidates: int = 24,
) -> np.ndarray:
    """The exact distance of ``points`` (n, d) to an interface given as facets (m, d, d: segments in 2D, triangles in
    3D), with the sign of ``signs`` (n: -1 in the solid, +1 in the void, 0 on the interface). On MPI ranks (``comm``,
    an mpi4py communicator) every rank passes its own facets and gets the distance to all ranks' facets: the interface
    is gathered whole on every rank, the points stay where they are. Each point takes the nearest of the facets whose
    centroids are its ``candidates`` nearest. Without facets anywhere the result is ``signs`` with 0 as -1."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    d = points.shape[1]
    facets = np.asarray(facets, dtype=np.float64).reshape(-1, d, d)
    if comm is not None and comm.size > 1:
        facets = np.concatenate(comm.allgather(facets), axis=0)
    signs = np.asarray(signs, dtype=np.float64)
    if len(facets) == 0:
        return np.where(signs > 0.0, 1.0, -1.0)
    centroids = facets.mean(axis=1)
    k = min(int(candidates), len(facets))
    _, near = cKDTree(centroids).query(points, k=k)
    near = np.asarray(near).reshape(len(points), k)
    best = np.full(len(points), np.inf)
    for j in range(k):
        f = facets[near[:, j]]
        if d == 2:
            dist = _segment_distances(points, f[:, 0], f[:, 1])
        else:
            dist = _triangle_distances(points, f[:, 0], f[:, 1], f[:, 2])
        best = np.minimum(best, dist)
    return np.where(signs > 0.0, best, np.where(signs < 0.0, -best, 0.0))
