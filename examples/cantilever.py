#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the circular-hole cantilever example from sotuto/base."""

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


CANTILEVER_LOAD_Y_MIN = 0.45
CANTILEVER_LOAD_Y_MAX = 0.55
CANTILEVER_HOLE_RADIUS = 0.05
CANTILEVER_HOLE_ROWS = (
    (6, 0.285, 0.285, 1.00),
    (5, 0.3333, 0.3333, 0.75),
    (6, 0.285, 0.285, 0.50),
    (5, 0.3333, 0.3333, 0.25),
    (6, 0.285, 0.285, 0.00),
)

CANTILEVER_PARAMETERS = Parameters(
    length=2.0,
    height=1.0,
    initial_left_segments=100,
    initial_bottom_segments=200,
    initial_top_segments=200,
    lambda_lame=0.5769,
    mu=0.3846,
    load_x=0.0,
    load_y=-1.0,
    regularization=0.02,
    ref_dirichlet=1,
    ref_dirichlet_roller=4,
    ref_neumann=2,
    ref_void=2,
    ref_solid=3,
    ref_interface=10,
    roller_support_required=False,
    mmg_binary="mmg2d_O3",
    mmg_hmin=0.001,
    mmg_hmax=0.02,
    mmg_hausd=0.001,
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
    volume_target=0.7,
    mesh_size_value=0.02,
    eps=1.0e-6,
    normalization_update_iteration=50,
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
    """Return the circular-hole cantilever level set from sotuto/base."""

    x = coords[:, 0]
    y = coords[:, 1]
    distance = np.full_like(x, 10.0, dtype=float)
    for count, start, spacing, cy in CANTILEVER_HOLE_ROWS:
        for i in range(count):
            cx = start + i * spacing
            distance = np.minimum(
                distance,
                np.hypot(x - cx, y - cy) - CANTILEVER_HOLE_RADIUS,
            )
    return -distance


def mark_boundaries(msh: mesh.Mesh, params: Parameters) -> BoundaryData:
    """Mark the cantilever clamp and right-edge load patch."""

    fdim = msh.topology.dim - 1
    left = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    right = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], params.length)
    )
    right_midpoints = mesh.compute_midpoints(msh, fdim, right)
    in_load_patch = np.logical_and(
        right_midpoints[:, 1] >= CANTILEVER_LOAD_Y_MIN - 1.0e-12,
        right_midpoints[:, 1] <= CANTILEVER_LOAD_Y_MAX + 1.0e-12,
    )
    load = right[in_load_patch]
    if len(load) == 0:
        raise RuntimeError("No load facets found for the cantilever example.")

    return BoundaryData(
        left_facets=left.astype(np.int32),
        load_facets=load.astype(np.int32),
        facet_tags=facet_tags_from_groups(
            msh,
            fdim,
            [
                (left, params.ref_dirichlet),
                (load, params.ref_neumann),
            ],
        ),
    )


if __name__ == "__main__":
    run_example(
        CANTILEVER_PARAMETERS,
        initial_level_set_values,
        mark_boundaries,
        Path("results_cantilever"),
        "Circular-hole cantilever compliance optimization.",
    )
