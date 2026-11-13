# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Topological derivative hole creation for fitted level-set optimization."""

from __future__ import annotations

import math

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from scipy.spatial import cKDTree

from .mechanics import SolidSubmesh, State, elastic_energy_density
from .optimization import DescentData
from . import solvers
from .parameters import Parameters


def _cell_vertices(msh: mesh.Mesh) -> np.ndarray:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    num_cells = msh.topology.index_map(tdim).size_local
    return np.asarray([c_to_v.links(c) for c in range(num_cells)], dtype=np.int64)


def _vertex_values(function: fem.Function) -> np.ndarray:
    """Return scalar P1 values in mesh-vertex order."""

    V = function.function_space
    if V.dofmap.bs != 1:
        raise RuntimeError("Expected a scalar P1 function.")
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, dofs = cKDTree(coords).query(vertex_coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError(f"Could not map {function.name} dofs to mesh vertices.")
    return np.asarray(function.x.array[dofs], dtype=np.float64)


def _set_vertex_values(function: fem.Function, values: np.ndarray) -> None:
    """Assign scalar P1 values provided in mesh-vertex order."""

    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, vertices = cKDTree(vertex_coords).query(coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError(f"Could not map mesh vertices to {function.name} dofs.")
    function.x.array[:] = np.asarray(values, dtype=np.float64)[vertices]
    function.x.scatter_forward()


def _interpolate_submesh_scalar_to_parent(
    source: fem.Function,
    target_mesh: mesh.Mesh,
    cell_map: mesh.EntityMap,
    name: str,
) -> fem.Function:
    """Interpolate a fitted-submesh scalar P1 function onto the parent mesh."""

    target = fem.Function(fem.functionspace(target_mesh, ("Lagrange", 1)), name=name)
    target.x.array[:] = 0.0
    tdim = source.function_space.mesh.topology.dim
    sub_cells = np.arange(
        source.function_space.mesh.topology.index_map(tdim).size_local,
        dtype=np.int32,
    )
    parent_cells = cell_map.sub_topology_to_topology(sub_cells, inverse=False)
    target.interpolate(source, cells0=sub_cells, cells1=parent_cells)
    target.x.scatter_forward()
    return target


def topological_compliance_derivative(
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    parent_mesh: mesh.Mesh,
    params: Parameters,
) -> fem.Function:
    """Return the P1 compliance topological derivative from ``topgradCp.edp``."""

    solid_mesh = solid_domain.mesh
    V = fem.functionspace(solid_mesh, ("Lagrange", 1))
    g = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=solid_mesh)

    div_u = ufl.div(solid_displacement)
    energy = elastic_energy_density(solid_displacement, params)
    trace_stress = (2.0 * params.mu + 2.0 * params.lambda_lame) * div_u
    coefficient = math.pi * (params.lambda_lame + 2.0 * params.mu) / (
        2.0 * params.mu * (params.lambda_lame + params.mu)
    )
    derivative_density = coefficient * (
        4.0 * params.mu * energy
        + (params.lambda_lame - params.mu) * trace_stress * div_u
    )

    problem = petsc.LinearProblem(
        g * v * dx,
        derivative_density * v * dx,
        petsc_options_prefix="sotutox_topological_compliance_",
        petsc_options=solvers.options("mass", solid_mesh),
    )
    on_solid = problem.solve()
    on_solid.name = "topological_compliance"
    on_solid.x.scatter_forward()
    return _interpolate_submesh_scalar_to_parent(
        on_solid,
        parent_mesh,
        solid_domain.cell_map,
        "topological_compliance",
    )


def topological_volume_derivative(msh: mesh.Mesh) -> fem.Function:
    """Return the P1 volume topological derivative from ``topgradV.edp``."""

    derivative = fem.Function(
        fem.functionspace(msh, ("Lagrange", 1)),
        name="topological_volume",
    )
    derivative.x.array[:] = -math.pi
    derivative.x.scatter_forward()
    return derivative


def topological_merit_derivative(
    topological_compliance: fem.Function,
    topological_volume: fem.Function,
    state: State,
    descent: DescentData,
    params: Parameters,
) -> fem.Function:
    """Combine topological derivatives as in ``descentTG.edp``."""

    derivative = fem.Function(
        topological_compliance.function_space,
        name="topological_merit",
    )
    volume_coefficient = (
        -descent.alpha_objective * descent.lagrange
        + descent.alpha_constraint
        / descent.penalty
        * (state.volume - params.target_volume)
    )
    derivative.x.array[:] = (
        descent.alpha_objective * topological_compliance.x.array
        + volume_coefficient * topological_volume.x.array
    )
    derivative.x.scatter_forward()
    return derivative


def _polygon_area(points: list[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    coords = np.asarray(points, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.dot(coords[:, 0], np.roll(coords[:, 1], -1))
            - np.dot(coords[:, 1], np.roll(coords[:, 0], -1))
        )
    )


def _negative_triangle_area(points: np.ndarray, values: np.ndarray) -> float:
    """Area of the linear P1 subregion where ``values <= 0`` in one triangle."""

    if np.all(values <= 0.0):
        a = points[1] - points[0]
        b = points[2] - points[0]
        return 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))
    if np.all(values >= 0.0):
        return 0.0

    clipped: list[np.ndarray] = []
    for i in range(3):
        p0 = points[i]
        p1 = points[(i + 1) % 3]
        v0 = values[i]
        v1 = values[(i + 1) % 3]
        inside0 = v0 <= 0.0
        inside1 = v1 <= 0.0
        if inside0:
            clipped.append(p0)
        if inside0 != inside1:
            t = v0 / (v0 - v1)
            clipped.append(p0 + t * (p1 - p0))
    return _polygon_area(clipped)


def _triangle_area(points: np.ndarray) -> float:
    a = points[1] - points[0]
    b = points[2] - points[0]
    return 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))


def _positive_triangle_area(points: np.ndarray, values: np.ndarray) -> float:
    return _triangle_area(points) - _negative_triangle_area(points, values)


def _connected_vertex_components(
    triangles: np.ndarray, vertex_mask: np.ndarray
) -> list[np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(len(vertex_mask))]
    for triangle in triangles:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = int(triangle[i])
            b = int(triangle[j])
            if vertex_mask[a] and vertex_mask[b]:
                adjacency[a].append(b)
                adjacency[b].append(a)

    seen = np.zeros(len(vertex_mask), dtype=bool)
    components: list[np.ndarray] = []
    for start in np.flatnonzero(vertex_mask):
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _prune_small_hole_components(
    phi_values: np.ndarray,
    candidate: np.ndarray,
    msh: mesh.Mesh,
    min_radius: float,
) -> np.ndarray:
    if min_radius <= 0.0:
        return candidate

    triangles = _cell_vertices(msh)
    new_hole_vertices = np.logical_and(phi_values < 0.0, candidate > 0.0)
    components = _connected_vertex_components(triangles, new_hole_vertices)
    if not components:
        return candidate

    component_id = np.full(len(phi_values), -1, dtype=np.int64)
    for index, component in enumerate(components):
        component_id[component] = index

    points = msh.geometry.x[:, : msh.geometry.dim]
    areas = np.zeros(len(components), dtype=np.float64)
    for triangle in triangles:
        ids = component_id[triangle]
        ids = np.unique(ids[ids >= 0])
        if len(ids) == 0:
            continue
        area = _positive_triangle_area(points[triangle, :2], candidate[triangle])
        if area <= 0.0:
            continue
        for index in ids:
            areas[index] += area / len(ids)

    min_area = math.pi * min_radius**2
    keep = areas >= min_area

    result = np.array(candidate, copy=True)
    reject_vertices = np.logical_and(component_id >= 0, ~keep[component_id])
    result[reject_vertices] = phi_values[reject_vertices]
    return result


def level_set_negative_area(phi: fem.Function, values: np.ndarray | None = None) -> float:
    """Compute the exact P1 negative-region area on a triangular mesh."""

    msh = phi.function_space.mesh
    if msh.comm.size != 1:
        raise RuntimeError("Topological hole creation is serial-only.")
    vertex_values = _vertex_values(phi) if values is None else np.asarray(values)
    points = msh.geometry.x[:, : msh.geometry.dim]
    triangles = _cell_vertices(msh)
    area = 0.0
    for triangle in triangles:
        area += _negative_triangle_area(
            points[triangle, :2],
            vertex_values[triangle],
        )
    return float(msh.comm.allreduce(area, op=MPI.SUM))


def create_topological_hole_level_set(
    phi: fem.Function,
    topological_derivative: fem.Function,
    params: Parameters,
) -> fem.Function:
    """Remove material where the topological merit derivative is most negative."""

    if phi.function_space.mesh.comm.size != 1:
        raise RuntimeError("Topological hole creation is serial-only.")
    phi_values = _vertex_values(phi)
    derivative_values = _vertex_values(topological_derivative)
    eligible = phi_values < -params.topological_min_distance_from_interface
    if not np.any(eligible):
        raise RuntimeError("No solid vertices eligible for topological hole creation.")

    eligible_derivative = derivative_values[eligible]
    lower = float(np.min(eligible_derivative))
    upper = float(np.max(eligible_derivative))
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        raise RuntimeError("Topological derivative has no usable variation.")

    initial_volume = level_set_negative_area(phi)
    target_volume = (1.0 - params.topological_volume_fraction) * initial_volume
    candidate = np.array(phi_values, copy=True)

    for _ in range(params.topological_bisection_iterations):
        threshold = 0.5 * (lower + upper)
        candidate = np.maximum(phi_values, threshold - derivative_values)
        candidate[~eligible] = phi_values[~eligible]
        volume = level_set_negative_area(phi, candidate)
        if volume < target_volume:
            upper = threshold
        else:
            lower = threshold
        if abs(volume - target_volume) < params.topological_volume_tolerance:
            break

    candidate = _prune_small_hole_components(
        phi_values,
        candidate,
        phi.function_space.mesh,
        params.topological_min_hole_radius,
    )

    result = fem.Function(phi.function_space, name="phi")
    _set_vertex_values(result, candidate)
    return result
