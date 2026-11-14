#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the multi-load bridge example from ``sotuto/constraints``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import mesh

try:
    from common import run_example
except ModuleNotFoundError:
    from examples.common import run_example
from sotutox.mechanics import BoundaryData, facet_tags_from_groups
from sotutox.parameters import Parameters
from sotutox.constraints import run_multiload_constraints


MULTILOAD_NUM_LOAD_CASES = 7
MULTILOAD_SUPPORT_WIDTH = 1.0
MULTILOAD_HOLE_RADIUS = 0.2
MULTILOAD_LOAD_REF = 11

MULTILOAD_BRIDGE_PARAMETERS = Parameters(
    length=10.0,
    height=2.0,
    initial_left_segments=40,
    initial_bottom_segments=200,
    initial_top_segments=200,
    lambda_lame=0.5769,
    mu=0.3846,
    load_x=0.0,
    load_y=-1.0,
    regularization=0.1,
    ref_dirichlet=1,
    ref_dirichlet_roller=4,
    ref_neumann=MULTILOAD_LOAD_REF,
    ref_void=2,
    ref_solid=3,
    ref_interface=10,
    roller_support_required=False,
    mmg_binary="mmg2d_O3",
    mmg_hmin=0.02,
    mmg_hmax=0.05,
    mmg_hausd=0.005,
    mmg_hgrad=1.3,
    max_iterations=150,
    max_line_search=3,
    max_coef=2.0,
    min_coef=0.02,
    line_search_increase=1.1,
    line_search_decrease=0.6,
    tolerance=0.02,
    late_tolerance_iteration=100,
    late_tolerance=0.005,
    objective_weight=1.0,
    constraint_weight=0.1,
    volume_target=5.0,
    mesh_size_value=0.1,
    eps=1.0e-6,
    normalization_update_iteration=None,
    advection_backend="supg",
    supg_tau_scale=1.0,
    topological_derivatives=False,
    topological_interval=5,
    topological_last_iteration=50,
    topological_volume_fraction=0.01,
    topological_tolerance=0.05,
    topological_volume_tolerance=1.0e-6,
    topological_bisection_iterations=50,
    topological_min_distance_from_interface=0.0,
    topological_min_hole_radius=0.0,
    void_cleanup_min_hole_radius=0.0,
    output_every=1,
)


def initial_level_set_values(coords: np.ndarray) -> np.ndarray:
    """Return the perforated bridge level set from sotuto/constraints."""

    x = coords[:, 0]
    y = coords[:, 1]
    distance = np.full_like(x, 10.0, dtype=float)

    for i in range(11):
        distance = np.minimum(
            distance,
            np.hypot(x - i, y - 1.5) - MULTILOAD_HOLE_RADIUS,
        )
    for i in range(10):
        distance = np.minimum(
            distance,
            np.hypot(x - (0.5 + i), y - 1.0) - MULTILOAD_HOLE_RADIUS,
        )
    for i in range(11):
        distance = np.minimum(
            distance,
            np.hypot(x - i, y - 0.5) - MULTILOAD_HOLE_RADIUS,
        )

    bottom_cut = np.logical_and(x > 1.0, x < 9.0)
    distance[bottom_cut] = np.minimum(distance[bottom_cut], y[bottom_cut] - 0.1)
    return -distance


def _load_intervals(params: Parameters) -> list[tuple[float, float]]:
    segment_length = params.length / (2 * MULTILOAD_NUM_LOAD_CASES - 1)
    intervals: list[tuple[float, float]] = []
    for k in range(2 * MULTILOAD_NUM_LOAD_CASES - 1):
        if k % 2 != 0:
            continue
        right = params.length - k * segment_length
        left = right - segment_length
        intervals.append((left, right))
    return intervals


def mark_boundaries(msh: mesh.Mesh, params: Parameters) -> BoundaryData:
    """Mark the two bottom supports and all top load patches."""

    fdim = msh.topology.dim - 1
    bottom = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    top = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], params.height)
    )

    bottom_midpoints = mesh.compute_midpoints(msh, fdim, bottom)
    bottom_x = bottom_midpoints[:, 0]
    eps = 1.0e-12
    supports = bottom[
        np.logical_or(
            bottom_x <= MULTILOAD_SUPPORT_WIDTH + eps,
            bottom_x >= params.length - MULTILOAD_SUPPORT_WIDTH - eps,
        )
    ]

    top_midpoints = mesh.compute_midpoints(msh, fdim, top)
    top_x = top_midpoints[:, 0]
    load_groups: list[tuple[np.ndarray, int]] = []
    for i, (left, right) in enumerate(_load_intervals(params)):
        in_load = np.logical_and(top_x >= left - eps, top_x <= right + eps)
        load_group = top[in_load]
        if len(load_group) > 0:
            load_groups.append((load_group, MULTILOAD_LOAD_REF + i))

    if msh.comm.allreduce(len(supports)) == 0:
        raise RuntimeError("No support facets found for the multiload bridge.")
    if len(load_groups) != MULTILOAD_NUM_LOAD_CASES:
        raise RuntimeError("No load facets found for the multiload bridge.")

    return BoundaryData(
        left_facets=supports.astype(np.int32),
        load_facets=np.concatenate([group for group, _ in load_groups]).astype(
            np.int32
        ),
        facet_tags=facet_tags_from_groups(
            msh,
            fdim,
            [(supports, params.ref_dirichlet), *load_groups],
        ),
    )


if __name__ == "__main__":
    load_refs = [MULTILOAD_LOAD_REF + i for i in range(MULTILOAD_NUM_LOAD_CASES)]
    loads = [(0.0, -1.0) for _ in range(MULTILOAD_NUM_LOAD_CASES)]
    run_example(
        MULTILOAD_BRIDGE_PARAMETERS,
        initial_level_set_values,
        mark_boundaries,
        Path("results_multiload_bridge"),
        "Multi-load constrained bridge optimization from sotuto/constraints.",
        driver=run_multiload_constraints,
        run_kwargs={
            "load_refs": load_refs,
            "loads": loads,
            "compliance_threshold_factor": 0.9,
            "constraint_penalty": 2.0,
        },
    )
