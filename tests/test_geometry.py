# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from dataclasses import replace
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import bridge
from sotutox.mmg import RemeshedLevelSet
from sotutox.optimization_mmg import _small_void_component_cells


def test_bridge_initial_level_set_matches_topder_strip() -> None:
    coords = np.array(
        [
            [1.0, 0.55],
            [0.05, 0.05],
            [1.0, 0.05],
            [1.95, 0.05],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    phi = bridge.initial_level_set_values(coords)

    np.testing.assert_allclose(phi, coords[:, 1] - bridge.BRIDGE_INITIAL_LEVEL)


def test_bridge_defaults_enable_topological_derivatives() -> None:
    params = bridge.BRIDGE_PARAMETERS

    assert params.topological_derivatives
    assert params.topological_interval == 5
    assert params.topological_last_iteration == 50
    assert params.topological_volume_fraction == 0.01
    assert params.topological_min_distance_from_interface == 0.15
    assert params.topological_min_hole_radius == 0.04
    assert params.void_cleanup_min_hole_radius == 0.04


def test_small_void_cleanup_keeps_largest_component() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [3.0, 0.0],
            [3.05, 0.0],
            [3.0, 0.05],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=np.float64)
    )
    msh = mesh.create_mesh(MPI.COMM_WORLD, cells, domain, points)
    cell_tags = mesh.meshtags(
        msh,
        msh.topology.dim,
        np.array([0, 1], dtype=np.int32),
        np.array([bridge.BRIDGE_PARAMETERS.ref_void] * 2, dtype=np.int32),
    )
    facet_tags = mesh.meshtags(
        msh,
        msh.topology.dim - 1,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )
    phi = fem.Function(fem.functionspace(msh, ("Lagrange", 1)), name="phi")
    params = replace(bridge.BRIDGE_PARAMETERS, void_cleanup_min_hole_radius=0.04)

    removed_cells, removed_areas = _small_void_component_cells(
        RemeshedLevelSet(msh, phi, cell_tags, facet_tags),
        params,
    )

    np.testing.assert_array_equal(removed_cells, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(removed_areas, [0.00125])
