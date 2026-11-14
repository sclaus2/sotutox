#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the 3D cantilever example from ``sotuto/base3d``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

try:
    from common import run_example
except ModuleNotFoundError:
    from examples.common import run_example
from sotutox.mechanics import BoundaryData, facet_tags_from_groups
from sotutox.parameters import Parameters


CANTILEVER_3D_DEPTH = 1.0
CANTILEVER_3D_HOLE_RADIUS = 0.05
CANTILEVER_3D_LOAD_CENTER = (0.5, 0.5)
CANTILEVER_3D_LOAD_RADIUS = 0.1

CANTILEVER_3D_PARAMETERS = Parameters(
    length=2.0,
    height=1.0,
    initial_left_segments=10,
    initial_bottom_segments=20,
    initial_top_segments=10,
    lambda_lame=0.5769,
    mu=0.3846,
    load_x=0.0,
    load_y=0.0,
    regularization=0.07,
    ref_dirichlet=1,
    ref_dirichlet_roller=4,
    ref_neumann=2,
    ref_void=2,
    ref_solid=3,
    ref_interface=10,
    roller_support_required=False,
    mmg_binary="mmg3d_O3",
    mmg_hmin=0.04,
    mmg_hmax=0.07,
    mmg_hausd=0.005,
    mmg_hgrad=1.4,
    max_iterations=200,
    max_line_search=3,
    max_coef=1.0,
    min_coef=0.02,
    line_search_increase=1.1,
    line_search_decrease=0.6,
    tolerance=0.01,
    late_tolerance_iteration=100,
    late_tolerance=0.005,
    objective_weight=1.0,
    constraint_weight=0.4,
    volume_target=0.45,
    mesh_size_value=0.05,
    eps=1.0e-6,
    normalization_update_iteration=200,
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
    load_z=-1.0,
)


def create_initial_mesh(params: Parameters) -> mesh.Mesh:
    """Create the rectangular tetrahedral hold-all used by ``base3d``."""

    return mesh.create_box(
        MPI.COMM_WORLD,
        (
            (0.0, 0.0, 0.0),
            (params.length, params.height, CANTILEVER_3D_DEPTH),
        ),
        (
            params.initial_bottom_segments,
            params.initial_left_segments,
            params.initial_top_segments,
        ),
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def initial_level_set_values(coords: np.ndarray) -> np.ndarray:
    """Return the spherical-hole level set from ``sotuto/base3d``."""

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    distance = np.full_like(x, 100.0, dtype=float)

    for k in range(3):
        for i in range(5):
            distance = np.minimum(
                distance,
                np.sqrt((x - i * 0.5) ** 2 + y**2 + (z - 0.5 * k) ** 2)
                - CANTILEVER_3D_HOLE_RADIUS,
            )

    for k in range(2):
        for i in range(4):
            distance = np.minimum(
                distance,
                np.sqrt(
                    (x - (0.125 + i * 0.5)) ** 2
                    + (y - 0.25) ** 2
                    + (z - (0.25 + 0.5 * k)) ** 2
                )
                - CANTILEVER_3D_HOLE_RADIUS,
            )

    for k in range(2):
        for i in range(5):
            distance = np.minimum(
                distance,
                np.sqrt((x - i * 0.5) ** 2 + (y - 0.5) ** 2 + (z - k) ** 2)
                - CANTILEVER_3D_HOLE_RADIUS,
            )

    for i in range(4):
        distance = np.minimum(
            distance,
            np.sqrt(
                (x - i * 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
            )
            - CANTILEVER_3D_HOLE_RADIUS,
        )

    for k in range(2):
        for i in range(4):
            distance = np.minimum(
                distance,
                np.sqrt(
                    (x - (0.125 + i * 0.5)) ** 2
                    + (y - 0.75) ** 2
                    + (z - (0.25 + 0.5 * k)) ** 2
                )
                - CANTILEVER_3D_HOLE_RADIUS,
            )

    for k in range(3):
        for i in range(5):
            distance = np.minimum(
                distance,
                np.sqrt(
                    (x - i * 0.5) ** 2
                    + (y - 1.0) ** 2
                    + (z - 0.5 * k) ** 2
                )
                - CANTILEVER_3D_HOLE_RADIUS,
            )

    return -distance


def mark_boundaries(msh: mesh.Mesh, params: Parameters) -> BoundaryData:
    """Mark the clamp at ``x=0`` and circular load patch at ``x=2``."""

    fdim = msh.topology.dim - 1
    clamp = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    right = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], params.length)
    )
    midpoints = mesh.compute_midpoints(msh, fdim, right)
    radius2 = (
        (midpoints[:, 1] - CANTILEVER_3D_LOAD_CENTER[0]) ** 2
        + (midpoints[:, 2] - CANTILEVER_3D_LOAD_CENTER[1]) ** 2
    )
    load = right[radius2 <= CANTILEVER_3D_LOAD_RADIUS**2 + 1.0e-12]

    if msh.comm.allreduce(len(clamp)) == 0:
        raise RuntimeError("No clamp facets found for the 3D cantilever.")
    if msh.comm.allreduce(len(load)) == 0:
        raise RuntimeError("No load facets found for the 3D cantilever.")

    return BoundaryData(
        left_facets=clamp.astype(np.int32),
        load_facets=load.astype(np.int32),
        facet_tags=facet_tags_from_groups(
            msh,
            fdim,
            [
                (clamp, params.ref_dirichlet),
                (load, params.ref_neumann),
            ],
        ),
    )


def shapeforest_settings(params: Parameters) -> dict:
    """``--remesher shapeforest``: the clamp x = 0, the load patch on x = 2 -- the square [0.4, 0.6]^2 that the first mesh's
    facets give the load disc, as mmg3d keeps it -- and solid parts must reach the clamp (mmg3d's -rmc). The sizes are
    those of the runs that reached mmg3d's design: h 0.07, tol h / 4, h_min 0.04, P1+G, the floor 0.5 h_min."""
    import shapeforest as sf

    load = sf.box(size=(0.5, 0.2, 0.2), center=(2.0, 0.5, 0.5), name="load")
    clamp = sf.box(size=(2.0, 3.0, 3.0), center=(-1.0 + 1e-7, 0.5, 0.5), name="clamp")
    return dict(
        patches=[(load, params.ref_neumann), (clamp, params.ref_dirichlet)],
        anchors={0: "clamp"},
        min_part=2e-5,
        h=params.mmg_hmax,
        tol=0.25 * params.mmg_hmax,
        h_min=params.mmg_hmin,
        representation="p1g",
        cross_floor=0.5,
    )


if __name__ == "__main__":
    run_example(
        CANTILEVER_3D_PARAMETERS,
        initial_level_set_values,
        mark_boundaries,
        Path("results_cantilever_3d"),
        "3D cantilever compliance optimization from sotuto/base3d.",
        create_initial_mesh=create_initial_mesh,
        shapeforest_settings=shapeforest_settings,
    )

