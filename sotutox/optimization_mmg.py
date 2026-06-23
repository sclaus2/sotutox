# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MMG-remeshed optimization loop for fitted level-set examples.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import meshio
import numpy as np
from dolfinx import fem, io, mesh
from scipy.spatial import cKDTree

from .advection import advect_level_set, normal_velocity_from_speed
from .geometry import (
    create_initial_rectangle_mesh,
    initialize_level_set,
    scalar_space,
)
from .mechanics import (
    BoundaryData,
    Gradients,
    SolidSubmesh,
    boundary_data_from_tags,
    compute_shape_gradients_fitted,
    evaluate_state_fitted,
)
from .mmg import (
    RemeshedLevelSet,
    distance_level_set_from_tags,
    run_mmg_level_set,
    run_mmg_standard,
)
from .optimization import compute_descent, merit
from .optimization import DescentData
from .parameters import Parameters
from .topology import (
    create_topological_hole_level_set,
    topological_compliance_derivative,
    topological_merit_derivative,
    topological_volume_derivative,
)


def _history_columns() -> list[str]:
    return [
        "iteration",
        "line_search",
        "accepted",
        "coef",
        "cells",
        "vertices",
        "compliance",
        "volume",
        "volume_error",
        "merit",
        "lagrange",
        "penalty",
        "alpha_objective",
        "alpha_constraint",
    ]


def _write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_history_columns())
        writer.writeheader()
        writer.writerows(rows)


def _cell_tag_function(
    msh: mesh.Mesh, cell_tags: mesh.MeshTags, name: str
) -> fem.Function:
    """Return a DG0 scalar copy of cell references for ParaView coloring."""

    Q = fem.functionspace(msh, ("DG", 0))
    values = np.zeros(msh.topology.index_map(msh.topology.dim).size_local)
    values[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
        cell_tags.values, dtype=np.float64
    )

    function = fem.Function(Q, name=name)
    for cell, value in enumerate(values):
        function.x.array[Q.dofmap.cell_dofs(cell)] = value
    function.x.scatter_forward()
    return function


def _cell_vertices(msh: mesh.Mesh) -> np.ndarray:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    num_cells = msh.topology.index_map(tdim).size_local
    return np.asarray([c_to_v.links(c) for c in range(num_cells)], dtype=np.int64)


def _facet_vertices(msh: mesh.Mesh, facets: np.ndarray) -> np.ndarray:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f_to_v = msh.topology.connectivity(fdim, 0)
    return np.asarray([f_to_v.links(int(f)) for f in facets], dtype=np.int64)


def _points3(msh: mesh.Mesh) -> np.ndarray:
    points = msh.geometry.x[:, : msh.geometry.dim]
    if points.shape[1] == 3:
        return points
    result = np.zeros((points.shape[0], 3), dtype=points.dtype)
    result[:, : points.shape[1]] = points
    return result


def _cell_tag_values(msh: mesh.Mesh, cell_tags: mesh.MeshTags) -> np.ndarray:
    values = np.zeros(msh.topology.index_map(msh.topology.dim).size_local)
    values[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
        cell_tags.values, dtype=np.float64
    )
    return values


def _dg0_cell_values(function: fem.Function) -> np.ndarray:
    Q = function.function_space
    values = np.zeros(Q.mesh.topology.index_map(Q.mesh.topology.dim).size_local)
    for cell in range(len(values)):
        values[cell] = float(function.x.array[Q.dofmap.cell_dofs(cell)][0])
    return values


def _vertex_values(function: fem.Function) -> np.ndarray:
    V = function.function_space
    bs = V.dofmap.bs
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    points = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, dofs = cKDTree(coords).query(points)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError(f"Could not map {function.name} dofs to mesh vertices.")

    values = function.x.array.reshape((-1, bs))[dofs]
    if bs == 1:
        return values[:, 0]
    if bs == 3:
        return values
    result = np.zeros((values.shape[0], 3), dtype=values.dtype)
    result[:, :bs] = values
    return result


def _set_vertex_values(function: fem.Function, values: np.ndarray) -> None:
    """Assign scalar P1 values provided in mesh-vertex order."""

    V = function.function_space
    if V.dofmap.bs != 1:
        raise RuntimeError("Expected a scalar P1 function.")
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    points = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, vertices = cKDTree(points).query(coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError(f"Could not map mesh vertices to {function.name} dofs.")
    function.x.array[:] = np.asarray(values, dtype=np.float64)[vertices]
    function.x.scatter_forward()


def _line_tag_values(tags: mesh.MeshTags) -> tuple[np.ndarray, np.ndarray]:
    facets = np.asarray(tags.indices, dtype=np.int32)
    values = np.asarray(tags.values, dtype=np.float64)
    return facets, values


def _write_pvd(path: Path, entries: list[tuple[int, Path]]) -> None:
    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for iteration, file_path in entries:
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(float(iteration)),
            group="",
            part="0",
            file=file_path.as_posix(),
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_parent_vtu(
    path: Path,
    step: RemeshedLevelSet,
    solid: fem.Function,
    state,
    gradients: Gradients | None = None,
    velocity: fem.Function | None = None,
) -> None:
    cells = _cell_vertices(step.mesh)
    facets, facet_refs = _line_tag_values(step.facet_tags)
    facet_cells = _facet_vertices(step.mesh, facets)
    cell_type = "tetra" if step.mesh.topology.dim == 3 else "triangle"
    facet_type = "triangle" if step.mesh.topology.dim == 3 else "line"
    num_facets = len(facet_cells)
    num_cells = len(cells)

    point_data = {
        "phi": _vertex_values(step.phi),
        "displacement": _vertex_values(state.displacement),
    }
    if gradients is not None:
        point_data["grad_compliance"] = _vertex_values(gradients.compliance)
        point_data["grad_volume"] = _vertex_values(gradients.volume)
    if velocity is not None:
        point_data["descent_velocity"] = _vertex_values(velocity)

    cell_data = {
        "cell_reference": [
            _cell_tag_values(step.mesh, step.cell_tags),
            np.zeros(num_facets),
        ],
        "facet_reference": [
            np.zeros(num_cells),
            facet_refs,
        ],
        "solid_indicator": [
            _dg0_cell_values(solid),
            np.zeros(num_facets),
        ],
    }
    meshio.write(
        path,
        meshio.Mesh(
            _points3(step.mesh),
            [(cell_type, cells), (facet_type, facet_cells)],
            point_data=point_data,
            cell_data=cell_data,
        ),
    )


def _write_solid_vtu(
    path: Path, solid_domain: SolidSubmesh, solid_displacement: fem.Function
) -> None:
    cells = _cell_vertices(solid_domain.mesh)
    facets, facet_refs = _line_tag_values(solid_domain.facet_tags)
    facet_cells = _facet_vertices(solid_domain.mesh, facets)
    cell_type = "tetra" if solid_domain.mesh.topology.dim == 3 else "triangle"
    facet_type = "triangle" if solid_domain.mesh.topology.dim == 3 else "line"
    num_cells = len(cells)
    num_facets = len(facet_cells)

    meshio.write(
        path,
        meshio.Mesh(
            _points3(solid_domain.mesh),
            [(cell_type, cells), (facet_type, facet_cells)],
            point_data={"solid_displacement": _vertex_values(solid_displacement)},
            cell_data={
                "solid_facet_reference": [
                    np.zeros(num_cells),
                    facet_refs,
                ],
                "solid_cell": [
                    np.ones(num_cells),
                    np.zeros(num_facets),
                ],
            },
        ),
    )


def _write_paraview_step(
    output_dir: Path,
    iteration: int,
    step: RemeshedLevelSet,
    solid: fem.Function,
    state,
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    parent_entries: list[tuple[int, Path]],
    solid_entries: list[tuple[int, Path]],
    gradients: Gradients | None = None,
    velocity: fem.Function | None = None,
) -> None:
    vtu_dir = output_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    parent_path = vtu_dir / f"step_{iteration:04d}.vtu"
    solid_path = vtu_dir / f"step_{iteration:04d}_solid.vtu"
    _write_parent_vtu(parent_path, step, solid, state, gradients, velocity)
    _write_solid_vtu(solid_path, solid_domain, solid_displacement)

    parent_entries.append((iteration, parent_path.relative_to(output_dir)))
    solid_entries.append((iteration, solid_path.relative_to(output_dir)))
    _write_pvd(output_dir / "steps.pvd", parent_entries)
    _write_pvd(output_dir / "solid_steps.pvd", solid_entries)


def _write_meshtags(
    xdmf: io.XDMFFile, tags: mesh.MeshTags, geometry: mesh.Geometry, name: str
) -> None:
    tags.name = name
    xdmf.write_meshtags(tags, geometry)


def _write_step(
    path: Path,
    step: RemeshedLevelSet,
    solid: fem.Function,
    state,
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    gradients: Gradients | None = None,
    velocity: fem.Function | None = None,
) -> None:
    """Write one remeshed step. A changing mesh cannot share one XDMF grid."""

    with io.XDMFFile(step.mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(step.mesh)
        _write_meshtags(xdmf, step.cell_tags, step.mesh.geometry, "cell_reference")
        _write_meshtags(xdmf, step.facet_tags, step.mesh.geometry, "facet_reference")
        xdmf.write_function(_cell_tag_function(step.mesh, step.cell_tags, "cell_reference"))
        xdmf.write_function(step.phi)
        xdmf.write_function(solid)
        xdmf.write_function(state.displacement)
        if gradients is not None:
            xdmf.write_function(gradients.compliance)
            xdmf.write_function(gradients.volume)
        if velocity is not None:
            xdmf.write_function(velocity)

    solid_path = path.with_name(f"{path.stem}_solid.xdmf")
    with io.XDMFFile(solid_domain.mesh.comm, solid_path, "w") as xdmf:
        xdmf.write_mesh(solid_domain.mesh)
        _write_meshtags(
            xdmf,
            solid_domain.facet_tags,
            solid_domain.mesh.geometry,
            "solid_facet_reference",
        )
        xdmf.write_function(solid_displacement)


def _num_cells(step: RemeshedLevelSet) -> int:
    return step.mesh.topology.index_map(step.mesh.topology.dim).size_local


def _num_vertices(step: RemeshedLevelSet) -> int:
    return step.mesh.geometry.x.shape[0]


def _use_topological_derivative(iteration: int, params: Parameters) -> bool:
    if not params.topological_derivatives:
        return False
    if params.topological_interval <= 0:
        return False
    return (
        iteration % params.topological_interval == 0
        and iteration <= params.topological_last_iteration
    )


def _triangle_areas(msh: mesh.Mesh) -> np.ndarray:
    triangles = _cell_vertices(msh)
    points = msh.geometry.x[:, :2]
    edge_a = points[triangles[:, 1]] - points[triangles[:, 0]]
    edge_b = points[triangles[:, 2]] - points[triangles[:, 0]]
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    return 0.5 * np.abs(cross)


def _cell_tag_int_values(msh: mesh.Mesh, cell_tags: mesh.MeshTags) -> np.ndarray:
    values = np.full(
        msh.topology.index_map(msh.topology.dim).size_local,
        -1,
        dtype=np.int32,
    )
    values[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
        cell_tags.values, dtype=np.int32
    )
    return values


def _void_components(
    msh: mesh.Mesh, cell_tags: mesh.MeshTags, params: Parameters
) -> list[np.ndarray]:
    refs = _cell_tag_int_values(msh, cell_tags)
    void = refs == params.ref_void
    if not np.any(void):
        return []

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    visited = np.zeros(len(void), dtype=bool)
    components: list[np.ndarray] = []
    for seed in np.flatnonzero(void):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [int(seed)]
        component: list[int] = []
        while stack:
            cell = stack.pop()
            component.append(cell)
            for facet in c_to_f.links(cell):
                for neighbor in f_to_c.links(int(facet)):
                    if void[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))
        components.append(np.asarray(component, dtype=np.int32))
    return components


def _small_void_component_cells(
    step: RemeshedLevelSet, params: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    if params.void_cleanup_min_hole_radius <= 0.0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

    components = _void_components(step.mesh, step.cell_tags, params)
    if len(components) <= 1:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

    areas = _triangle_areas(step.mesh)
    component_areas = np.asarray([areas[cells].sum() for cells in components])
    protected = int(np.argmax(component_areas))
    min_area = math.pi * params.void_cleanup_min_hole_radius**2
    remove = [
        i
        for i, area in enumerate(component_areas)
        if i != protected and area < min_area
    ]
    if not remove:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

    removed_cells = np.concatenate([components[i] for i in remove]).astype(np.int32)
    removed_areas = component_areas[remove]
    return removed_cells, removed_areas


def _facet_tags_without_removed_interfaces(
    msh: mesh.Mesh,
    facet_tags: mesh.MeshTags | None,
    removed_cells: np.ndarray,
    params: Parameters,
) -> mesh.MeshTags | None:
    if facet_tags is None or len(removed_cells) == 0:
        return facet_tags

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    removed = np.zeros(msh.topology.index_map(tdim).size_local, dtype=bool)
    removed[removed_cells] = True

    facets = np.asarray(facet_tags.indices, dtype=np.int32)
    values = np.asarray(facet_tags.values, dtype=np.int32)
    keep = np.ones(len(facets), dtype=bool)
    for i, (facet, value) in enumerate(zip(facets, values)):
        if value != params.ref_interface:
            continue
        if np.any(removed[f_to_c.links(int(facet))]):
            keep[i] = False

    facets = facets[keep]
    values = values[keep]
    order = np.argsort(facets)
    return mesh.meshtags(msh, fdim, facets[order], values[order])


def _cell_tags_with_removed_void_as_solid(
    msh: mesh.Mesh,
    cell_tags: mesh.MeshTags,
    removed_cells: np.ndarray,
    params: Parameters,
) -> mesh.MeshTags:
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    values = _cell_tag_int_values(msh, cell_tags)
    values[values < 0] = params.ref_solid
    values[removed_cells] = params.ref_solid
    return mesh.meshtags(msh, tdim, cells, values.astype(np.int32))


def _remove_void_cells_from_level_set(
    step: RemeshedLevelSet, removed_cells: np.ndarray, params: Parameters
) -> fem.Function:
    values = np.asarray(_vertex_values(step.phi), dtype=np.float64).copy()
    vertices = np.unique(_cell_vertices(step.mesh)[removed_cells].reshape(-1))
    values[vertices] = -np.maximum(np.abs(values[vertices]), params.mmg_hmin)

    cleaned = fem.Function(step.phi.function_space, name="phi")
    _set_vertex_values(cleaned, values)
    return cleaned


def _cleanup_small_void_components(
    step: RemeshedLevelSet,
    params: Parameters,
    mmg_dir: Path,
    name: str,
) -> RemeshedLevelSet:
    removed_cells, removed_areas = _small_void_component_cells(step, params)
    if len(removed_cells) == 0:
        return step

    cleaned_phi = _remove_void_cells_from_level_set(step, removed_cells, params)
    cleaned_cell_tags = _cell_tags_with_removed_void_as_solid(
        step.mesh, step.cell_tags, removed_cells, params
    )
    cleaned_facets = _facet_tags_without_removed_interfaces(
        step.mesh, step.facet_tags, removed_cells, params
    )
    area_preview = ", ".join(f"{area:.3e}" for area in removed_areas[:5])
    if len(removed_areas) > 5:
        area_preview += ", ..."
    print(
        "  cleanup "
        f"removed {len(removed_areas)} small void component(s) "
        f"area=[{area_preview}]"
    )

    cleaned = run_mmg_level_set(
        step.mesh,
        cleaned_phi,
        cleaned_facets,
        cleaned_cell_tags,
        params,
        mmg_dir,
        f"{name}_cleanup",
    )
    distance_level_set_from_tags(
        cleaned.phi, cleaned.cell_tags, cleaned.facet_tags, params
    )
    return cleaned


InitialLevelSet = Callable[[np.ndarray], np.ndarray]
BoundaryMarker = Callable[[mesh.Mesh, Parameters], BoundaryData]
InitialMeshFactory = Callable[[Parameters], mesh.Mesh]
StateEvaluator = Callable[
    [mesh.Mesh, mesh.MeshTags, mesh.MeshTags, Parameters],
    tuple[object, fem.Function, SolidSubmesh, fem.Function],
]
GradientEvaluator = Callable[
    [mesh.Mesh, BoundaryData, SolidSubmesh, fem.Function, Parameters],
    Gradients,
]
DescentEvaluator = Callable[[Gradients, object, Parameters, float], DescentData]


def run(
    params: Parameters,
    output_dir: Path,
    *,
    initial_level_set: InitialLevelSet,
    mark_boundaries: BoundaryMarker,
    create_initial_mesh: InitialMeshFactory = create_initial_rectangle_mesh,
    evaluate_state: StateEvaluator = evaluate_state_fitted,
    compute_gradients: GradientEvaluator = compute_shape_gradients_fitted,
    compute_descent_data: DescentEvaluator = compute_descent,
    write_xdmf: bool = True,
) -> list[dict]:
    """Run the generic MMG-remeshed fitted-domain optimization loop."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mmg_dir = output_dir / "mmg"

    initial_mesh = create_initial_mesh(params)
    initial_boundaries = mark_boundaries(initial_mesh, params)
    background = run_mmg_standard(
        initial_mesh,
        initial_boundaries.facet_tags,
        None,
        params,
        mmg_dir,
        "initial_background",
    )
    V0 = scalar_space(background.mesh)
    phi0 = initialize_level_set(V0, initial_level_set)
    current = run_mmg_level_set(
        background.mesh,
        phi0,
        background.facet_tags,
        background.cell_tags,
        params,
        mmg_dir,
        "initial",
    )

    history: list[dict] = []
    parent_pvd_entries: list[tuple[int, Path]] = []
    solid_pvd_entries: list[tuple[int, Path]] = []
    coef = 1.0
    max_norm_xi_objective = 0.0

    for iteration in range(params.max_iterations + 1):
        distance_level_set_from_tags(
            current.phi, current.cell_tags, current.facet_tags, params
        )
        boundaries = boundary_data_from_tags(current.facet_tags, params)
        state, solid, solid_domain, solid_displacement = evaluate_state(
            current.mesh, current.cell_tags, current.facet_tags, params
        )

        print(
            f"it={iteration:03d} "
            f"cells={_num_cells(current):d} "
            f"J={state.compliance:.6e} "
            f"V={state.volume:.6e} "
            f"V-V*={state.volume - params.target_volume:+.3e}"
        )

        if iteration == params.max_iterations:
            if write_xdmf and iteration % params.output_every == 0:
                _write_step(
                    output_dir / f"step_{iteration:04d}.xdmf",
                    current,
                    solid,
                    state,
                    solid_domain,
                    solid_displacement,
                )
                _write_paraview_step(
                    output_dir,
                    iteration,
                    current,
                    solid,
                    state,
                    solid_domain,
                    solid_displacement,
                    parent_pvd_entries,
                    solid_pvd_entries,
                )
            break

        gradients = compute_gradients(
            current.mesh,
            boundaries,
            solid_domain,
            solid_displacement,
            params,
        )
        descent = compute_descent_data(
            gradients, state, params, max_norm_xi_objective
        )
        velocity = normal_velocity_from_speed(current.phi, descent.speed, params)
        if iteration == params.normalization_update_iteration:
            max_norm_xi_objective = descent.norm_xi_objective

        if write_xdmf and iteration % params.output_every == 0:
            _write_step(
                output_dir / f"step_{iteration:04d}.xdmf",
                current,
                solid,
                state,
                solid_domain,
                solid_displacement,
                gradients,
                velocity,
            )
            _write_paraview_step(
                output_dir,
                iteration,
                current,
                solid,
                state,
                solid_domain,
                solid_displacement,
                parent_pvd_entries,
                solid_pvd_entries,
                gradients,
                velocity,
            )

        current_merit = merit(state, descent, params)
        current_tolerance = params.line_search_tolerance(iteration)

        if _use_topological_derivative(iteration, params):
            top_compliance = topological_compliance_derivative(
                solid_domain,
                solid_displacement,
                current.mesh,
                params,
            )
            top_volume = topological_volume_derivative(current.mesh)
            top_merit = topological_merit_derivative(
                top_compliance,
                top_volume,
                state,
                descent,
                params,
            )
            raw_trial_phi = create_topological_hole_level_set(
                current.phi,
                top_merit,
                params,
            )
            accepted = False
            trial_merit = current_merit
            trial_state = state
            try:
                trial = run_mmg_level_set(
                    current.mesh,
                    raw_trial_phi,
                    current.facet_tags,
                    current.cell_tags,
                    params,
                    mmg_dir,
                    f"it{iteration:04d}_topology",
                )
                trial = _cleanup_small_void_components(
                    trial, params, mmg_dir, f"it{iteration:04d}_topology"
                )
                distance_level_set_from_tags(
                    trial.phi, trial.cell_tags, trial.facet_tags, params
                )
                trial_state, _, _, _ = evaluate_state(
                    trial.mesh, trial.cell_tags, trial.facet_tags, params
                )
                trial_merit = merit(trial_state, descent, params)
                accepted = (
                    math.isfinite(trial_state.compliance)
                    and math.isfinite(trial_state.volume)
                    and math.isfinite(trial_merit)
                    and trial_merit
                    < current_merit
                    + params.topological_tolerance * abs(current_merit)
                )
            except RuntimeError as exc:
                print(
                    "  topder "
                    f"volume_fraction={params.topological_volume_fraction:.3e} "
                    "failed; reject"
                )
                trial = current
                trial_state = state
                trial_merit = current_merit
                accepted = False
                if iteration == 0:
                    raise RuntimeError(
                        "Initial topological derivative update failed."
                    ) from exc

            print(
                f"  topder volume_fraction={params.topological_volume_fraction:.3e} "
                f"trial_merit={trial_merit:.6e} {'accept' if accepted else 'reject'}"
            )
            if accepted:
                current = trial
            row = {
                "iteration": iteration,
                "line_search": -1,
                "accepted": accepted,
                "coef": coef,
                "cells": _num_cells(current),
                "vertices": _num_vertices(current),
                "compliance": trial_state.compliance if accepted else state.compliance,
                "volume": trial_state.volume if accepted else state.volume,
                "volume_error": (
                    trial_state.volume - params.target_volume
                    if accepted
                    else state.volume - params.target_volume
                ),
                "merit": trial_merit if accepted else current_merit,
                "lagrange": descent.lagrange,
                "penalty": descent.penalty,
                "alpha_objective": descent.alpha_objective,
                "alpha_constraint": descent.alpha_constraint,
            }
            history.append(row)
            _write_history(output_dir / "history.csv", history)
            continue

        accepted = None
        accepted_state = None
        accepted_merit = None
        accepted_k = params.max_line_search - 1

        for k in range(params.max_line_search):
            raw_trial_phi = advect_level_set(
                current.phi,
                velocity,
                coef,
                params,
            )
            try:
                trial = run_mmg_level_set(
                    current.mesh,
                    raw_trial_phi,
                    current.facet_tags,
                    current.cell_tags,
                    params,
                    mmg_dir,
                    f"it{iteration:04d}_ls{k}",
                )
                trial = _cleanup_small_void_components(
                    trial, params, mmg_dir, f"it{iteration:04d}_ls{k}"
                )
            except RuntimeError as exc:
                print(
                    f"  ls={k} coef={coef:.3e} MMG remesh failed; reject"
                )
                if k == params.max_line_search - 1:
                    raise RuntimeError(
                        "Line search failed because MMG rejected every trial mesh."
                    ) from exc
                coef = max(params.min_coef, params.line_search_decrease * coef)
                continue
            distance_level_set_from_tags(
                trial.phi, trial.cell_tags, trial.facet_tags, params
            )
            try:
                trial_state, _, _, _ = evaluate_state(
                    trial.mesh, trial.cell_tags, trial.facet_tags, params
                )
            except RuntimeError as exc:
                print(
                    f"  ls={k} coef={coef:.3e} fitted state failed; reject"
                )
                if k == params.max_line_search - 1:
                    raise RuntimeError(
                        "Line search failed because every fitted trial state failed."
                    ) from exc
                coef = max(params.min_coef, params.line_search_decrease * coef)
                continue
            trial_merit = merit(trial_state, descent, params)
            if not (
                math.isfinite(trial_state.compliance)
                and math.isfinite(trial_state.volume)
                and math.isfinite(trial_merit)
            ):
                print(
                    f"  ls={k} coef={coef:.3e} non-finite fitted state; reject"
                )
                if k == params.max_line_search - 1:
                    raise RuntimeError(
                        "Line search failed because every fitted trial state was non-finite."
                    )
                coef = max(params.min_coef, params.line_search_decrease * coef)
                continue
            accept = (
                trial_merit < current_merit + current_tolerance * abs(current_merit)
                or k == params.max_line_search - 1
                or coef < params.min_coef
            )
            print(
                f"  ls={k} coef={coef:.3e} cells={_num_cells(trial):d} "
                f"trial_merit={trial_merit:.6e} {'accept' if accept else 'reject'}"
            )
            if accept:
                accepted = trial
                accepted_state = trial_state
                accepted_merit = trial_merit
                accepted_k = k
                coef = min(params.max_coef, params.line_search_increase * coef)
                break
            coef = max(params.min_coef, params.line_search_decrease * coef)

        if accepted is None:
            raise RuntimeError("Line search failed without producing a trial mesh.")

        current = accepted
        row = {
            "iteration": iteration,
            "line_search": accepted_k,
            "accepted": True,
            "coef": coef,
            "cells": _num_cells(current),
            "vertices": _num_vertices(current),
            "compliance": accepted_state.compliance,
            "volume": accepted_state.volume,
            "volume_error": accepted_state.volume - params.target_volume,
            "merit": accepted_merit,
            "lagrange": descent.lagrange,
            "penalty": descent.penalty,
            "alpha_objective": descent.alpha_objective,
            "alpha_constraint": descent.alpha_constraint,
        }
        history.append(row)
        _write_history(output_dir / "history.csv", history)

    _write_history(output_dir / "history.csv", history)
    return history
