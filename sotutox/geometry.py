# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Geometry and level-set utilities for the generic MMG optimization driver.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.spatial import cKDTree

from .parameters import Parameters
from .redistance import graph_signed_distance


def create_initial_rectangle_mesh(params: Parameters) -> mesh.Mesh:
    """Create the rectangular hold-all mesh requested by ``params``.

    DOLFINx creates a structured triangular rectangle. Examples choose the
    rectangle dimensions and boundary segment counts.
    """

    if params.initial_bottom_segments != params.initial_top_segments:
        raise ValueError("Bottom and top segment counts differ.")
    if params.initial_left_segments <= 0 or params.initial_bottom_segments <= 0:
        raise ValueError("Initial mesh segment counts must be positive.")

    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        ((0.0, 0.0), (params.length, params.height)),
        (params.initial_bottom_segments, params.initial_left_segments),
        cell_type=mesh.CellType.triangle,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def scalar_space(msh: mesh.Mesh) -> fem.FunctionSpace:
    """P1 scalar space used for the level set and scalar velocities."""

    return fem.functionspace(msh, ("Lagrange", 1))


def _mesh_edges(msh: mesh.Mesh) -> np.ndarray:
    """Return local mesh edges as pairs of geometry vertex indices."""

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f_to_v = msh.topology.connectivity(fdim, 0)
    num_facets = msh.topology.index_map(fdim).size_local
    return np.asarray(
        [f_to_v.links(facet) for facet in range(num_facets)], dtype=np.int64
    )


def _vertex_values(function: fem.Function) -> np.ndarray:
    """Return scalar P1 values in mesh-vertex order."""

    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, dofs = cKDTree(coords).query(vertex_coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match P1 dofs to mesh vertices.")
    return np.asarray(function.x.array[dofs], dtype=np.float64)


def _set_vertex_values(function: fem.Function, values: np.ndarray) -> None:
    """Assign scalar P1 values provided in mesh-vertex order."""

    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, vertices = cKDTree(vertex_coords).query(coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match mesh vertices to P1 dofs.")
    function.x.array[:] = np.asarray(values, dtype=np.float64)[vertices]
    function.x.scatter_forward()


def initialize_level_set(
    V: fem.FunctionSpace, values: Callable[[np.ndarray], np.ndarray]
) -> fem.Function:
    """Create the initial P1 level-set from an example-supplied function."""

    phi = fem.Function(V, name="phi")
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    phi.x.array[:] = np.asarray(values(coords), dtype=np.float64)
    phi.x.scatter_forward()
    return phi


def redistance(phi: fem.Function) -> None:
    """Replace ``phi`` by the native graph signed-distance approximation."""

    has_negative = np.any(phi.x.array < 0.0)
    has_positive = np.any(phi.x.array > 0.0)
    if not (has_negative and has_positive):
        return

    if phi.function_space.mesh.comm.size != 1:
        raise RuntimeError("Graph redistancing is serial-only.")
    msh = phi.function_space.mesh
    signed = graph_signed_distance(
        msh.geometry.x[:, : msh.geometry.dim],
        _mesh_edges(msh),
        _vertex_values(phi),
    )
    _set_vertex_values(phi, signed)
