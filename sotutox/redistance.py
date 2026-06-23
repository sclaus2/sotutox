# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dependency-light level-set redistancing helpers."""

from __future__ import annotations

import heapq

import numpy as np


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
    propagated with Dijkstra's algorithm over the mesh edge graph.
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

    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(points.shape[0])]

    for i_raw, j_raw in edges:
        i = int(i_raw)
        j = int(j_raw)
        length = float(np.linalg.norm(points[i] - points[j]))
        if length <= 0.0:
            continue
        adjacency[i].append((j, length))
        adjacency[j].append((i, length))

        vi = values[i]
        vj = values[j]
        abs_i = abs(vi)
        abs_j = abs(vj)
        if abs_i <= zero_tol:
            distances[i] = 0.0
        if abs_j <= zero_tol:
            distances[j] = 0.0
        if vi * vj < 0.0:
            denom = abs_i + abs_j
            if denom > zero_tol:
                distances[i] = min(distances[i], length * abs_i / denom)
                distances[j] = min(distances[j], length * abs_j / denom)

    queue: list[tuple[float, int]] = [
        (float(distance), vertex)
        for vertex, distance in enumerate(distances)
        if np.isfinite(distance)
    ]
    heapq.heapify(queue)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for neighbor, weight in adjacency[vertex]:
            trial = distance + weight
            if trial < distances[neighbor]:
                distances[neighbor] = trial
                heapq.heappush(queue, (trial, neighbor))

    fallback = np.abs(values)
    distances = np.where(np.isfinite(distances), distances, fallback)
    signs = np.where(values < 0.0, -1.0, 1.0)
    signed = signs * distances
    signed[np.abs(values) <= zero_tol] = 0.0
    return signed
