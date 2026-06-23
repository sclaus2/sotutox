#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the MBB-style bridge example from sotuto/topder."""

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


BRIDGE_INITIAL_LEVEL = 0.7
BRIDGE_LEFT_SUPPORT_END = 0.1
BRIDGE_LOAD_X_MIN = 0.95
BRIDGE_LOAD_X_MAX = 1.05
BRIDGE_ROLLER_X_MIN = 1.9

BRIDGE_PARAMETERS = Parameters(
    length=2.0,
    height=1.5,
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
    roller_support_required=True,
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
    topological_derivatives=True,
    topological_interval=5,
    topological_last_iteration=50,
    topological_volume_fraction=0.01,
    topological_tolerance=0.05,
    topological_volume_tolerance=1.0e-6,
    topological_bisection_iterations=50,
    topological_min_distance_from_interface=0.15,
    topological_min_hole_radius=0.04,
    void_cleanup_min_hole_radius=0.04,
    output_every=1,
)


def initial_level_set_values(coords: np.ndarray) -> np.ndarray:
    """Return the initial horizontal level set from sotuto/topder."""

    return coords[:, 1] - BRIDGE_INITIAL_LEVEL


def mark_boundaries(msh: mesh.Mesh, params: Parameters) -> BoundaryData:
    """Mark the MBB-style bottom supports and load patch."""

    fdim = msh.topology.dim - 1
    bottom = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    bottom_midpoints = mesh.compute_midpoints(msh, fdim, bottom)
    x = bottom_midpoints[:, 0]
    eps = 1.0e-12

    left_support = bottom[
        np.logical_and(x >= -eps, x <= BRIDGE_LEFT_SUPPORT_END + eps)
    ]
    load = bottom[
        np.logical_and(
            x >= BRIDGE_LOAD_X_MIN - eps,
            x <= BRIDGE_LOAD_X_MAX + eps,
        )
    ]
    roller = bottom[
        np.logical_and(x >= BRIDGE_ROLLER_X_MIN - eps, x <= params.length + eps)
    ]

    if len(left_support) == 0:
        raise RuntimeError("No left support facets found for the bridge example.")
    if len(load) == 0:
        raise RuntimeError("No load facets found for the bridge example.")
    if len(roller) == 0:
        raise RuntimeError("No roller support facets found for the bridge example.")

    return BoundaryData(
        left_facets=left_support.astype(np.int32),
        load_facets=load.astype(np.int32),
        roller_facets=roller.astype(np.int32),
        facet_tags=facet_tags_from_groups(
            msh,
            fdim,
            [
                (left_support, params.ref_dirichlet),
                (load, params.ref_neumann),
                (roller, params.ref_dirichlet_roller),
            ],
        ),
    )


if __name__ == "__main__":
    run_example(
        BRIDGE_PARAMETERS,
        initial_level_set_values,
        mark_boundaries,
        Path("results_bridge"),
        "MBB-style bridge compliance optimization.",
    )
