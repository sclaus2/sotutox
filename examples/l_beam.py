#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run the L-beam stress example from ``sotuto/stress``."""

from __future__ import annotations

from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import mesh
from mpi4py import MPI

try:
    from common import run_example
except ModuleNotFoundError:
    from examples.common import run_example
from sotutox.mechanics import BoundaryData, facet_tags_from_groups
from sotutox.parameters import Parameters
from sotutox.stress import (
    compute_stress_descent,
    compute_stress_shape_gradients_fitted,
    evaluate_stress_state_fitted,
)


L_BEAM_LOAD_Y_MIN = 0.45
L_BEAM_LOAD_Y_MAX = 0.55
L_BEAM_HOLE_RADIUS = 0.05
L_BEAM_STRESS_EXCLUDED_Y = 1.8
L_BEAM_STRESS_EXCLUDED_CENTER = (2.0, 0.5)
L_BEAM_STRESS_EXCLUDED_RADIUS = 0.1
L_BEAM_FIXED_RADIUS = 0.15
L_BEAM_FIXED_PENALTY = 1.0e16

L_BEAM_PARAMETERS = Parameters(
    length=2.0,
    height=2.0,
    initial_left_segments=160,
    initial_bottom_segments=160,
    initial_top_segments=160,
    lambda_lame=0.5769,
    mu=0.3846,
    load_x=0.0,
    load_y=-1.0,
    regularization=0.01,
    ref_dirichlet=1,
    ref_dirichlet_roller=4,
    ref_neumann=2,
    ref_void=2,
    ref_solid=3,
    ref_interface=10,
    roller_support_required=False,
    mmg_binary="mmg2d_O3",
    mmg_hmin=0.004,
    mmg_hmax=0.04,
    mmg_hausd=0.001,
    mmg_hgrad=1.3,
    max_iterations=200,
    max_line_search=3,
    max_coef=1.0,
    min_coef=0.02,
    line_search_increase=1.1,
    line_search_decrease=0.6,
    tolerance=0.02,
    late_tolerance_iteration=100,
    late_tolerance=0.005,
    objective_weight=1.0,
    constraint_weight=0.3,
    volume_target=0.7,
    mesh_size_value=0.02,
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
)


def create_initial_mesh(params: Parameters) -> mesh.Mesh:
    """Create a structured triangular mesh of the L-shaped hold-all."""

    nx = params.initial_bottom_segments
    ny = params.initial_left_segments
    xs = np.linspace(0.0, params.length, nx + 1)
    ys = np.linspace(0.0, params.height, ny + 1)
    points = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)

    def vertex(i: int, j: int) -> int:
        return j * (nx + 1) + i

    triangles: list[tuple[int, int, int]] = []
    for j in range(ny):
        cy = 0.5 * (ys[j] + ys[j + 1])
        for i in range(nx):
            cx = 0.5 * (xs[i] + xs[i + 1])
            if cx > 1.0 and cy > 1.0:
                continue
            v00 = vertex(i, j)
            v10 = vertex(i + 1, j)
            v01 = vertex(i, j + 1)
            v11 = vertex(i + 1, j + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    used = np.unique(np.asarray(triangles, dtype=np.int64).reshape(-1))
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    cells = remap[np.asarray(triangles, dtype=np.int64)]
    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=np.float64)
    )
    return mesh.create_mesh(MPI.COMM_WORLD, cells, domain, points[used])


def initial_level_set_values(coords: np.ndarray) -> np.ndarray:
    """Return the circular-hole L-beam level set from sotuto/stress."""

    x = coords[:, 0]
    y = coords[:, 1]
    distance = np.full_like(x, 100.0, dtype=float)

    rows = [
        (7, 0.0, 0.33333, 0.0),
        (7, 0.0, 0.33333, 1.0),
        (6, 0.0, 0.33333, 0.5),
        (6, 0.17, 0.33333, 0.25),
        (6, 0.17, 0.33333, 0.75),
        (3, 0.17, 0.33333, 1.25),
        (3, 0.17, 0.33333, 1.75),
        (4, 0.0, 0.33333, 1.5),
    ]
    for count, start, spacing, cy in rows:
        for i in range(count):
            cx = start + i * spacing
            distance = np.minimum(
                distance,
                np.hypot(x - cx, y - cy) - L_BEAM_HOLE_RADIUS,
            )
    return -distance


def stress_weight(msh: mesh.Mesh) -> ufl.core.expr.Expr:
    """Return the original ``kS`` localization from ``sotuto/stress``."""

    x = ufl.SpatialCoordinate(msh)
    circle = (
        (x[0] - L_BEAM_STRESS_EXCLUDED_CENTER[0]) ** 2
        + (x[1] - L_BEAM_STRESS_EXCLUDED_CENTER[1]) ** 2
    )
    return ufl.conditional(
        x[1] > L_BEAM_STRESS_EXCLUDED_Y,
        0.0,
        ufl.conditional(circle < L_BEAM_STRESS_EXCLUDED_RADIUS**2, 0.0, 1.0),
    )


def regularization_mass(msh: mesh.Mesh) -> ufl.core.expr.Expr:
    """Return ``1 + TGV*chifix`` from ``sotuto/stress/macros.idp``."""

    x = ufl.SpatialCoordinate(msh)
    circle = (
        (x[0] - L_BEAM_STRESS_EXCLUDED_CENTER[0]) ** 2
        + (x[1] - L_BEAM_STRESS_EXCLUDED_CENTER[1]) ** 2
    )
    chifix = ufl.conditional(circle < L_BEAM_FIXED_RADIUS**2, 1.0, 0.0)
    return 1.0 + L_BEAM_FIXED_PENALTY * chifix


def mark_boundaries(msh: mesh.Mesh, params: Parameters) -> BoundaryData:
    """Mark the L-beam top clamp and right-edge load patch."""

    fdim = msh.topology.dim - 1
    top = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: np.logical_and(np.isclose(x[1], params.height), x[0] <= 1.0),
    )
    right = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: np.logical_and(np.isclose(x[0], params.length), x[1] <= 1.0),
    )
    right_midpoints = mesh.compute_midpoints(msh, fdim, right)
    load = right[
        np.logical_and(
            right_midpoints[:, 1] >= L_BEAM_LOAD_Y_MIN - 1.0e-12,
            right_midpoints[:, 1] <= L_BEAM_LOAD_Y_MAX + 1.0e-12,
        )
    ]

    if len(top) == 0:
        raise RuntimeError("No Dirichlet facets found for the L-beam example.")
    if len(load) == 0:
        raise RuntimeError("No load facets found for the L-beam example.")

    return BoundaryData(
        left_facets=top.astype(np.int32),
        load_facets=load.astype(np.int32),
        facet_tags=facet_tags_from_groups(
            msh,
            fdim,
            [
                (top, params.ref_dirichlet),
                (load, params.ref_neumann),
            ],
        ),
    )


if __name__ == "__main__":
    run_example(
        L_BEAM_PARAMETERS,
        initial_level_set_values,
        mark_boundaries,
        Path("results_l_beam"),
        "L-beam stress optimization from sotuto/stress.",
        create_initial_mesh=create_initial_mesh,
        run_kwargs={
            "evaluate_state": lambda msh, cell_tags, facet_tags, params: (
                evaluate_stress_state_fitted(
                    msh, cell_tags, facet_tags, params, stress_weight
                )
            ),
            "compute_gradients": (
                lambda msh, boundaries, solid_domain, solid_displacement, params: (
                    compute_stress_shape_gradients_fitted(
                        msh,
                        boundaries,
                        solid_domain,
                        solid_displacement,
                        params,
                        stress_weight,
                        regularization_mass,
                    )
                )
            ),
            "compute_descent_data": (
                lambda gradients, state, params, max_norm: compute_stress_descent(
                    gradients,
                    state,
                    params,
                    max_norm,
                    regularization_mass,
                )
            ),
        },
    )
