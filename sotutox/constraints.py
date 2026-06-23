# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-load compliance-constraint driver for ``sotuto/constraints``."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from .advection import advect_level_set, normal_velocity_from_speed
from .geometry import create_initial_rectangle_mesh, initialize_level_set, scalar_space
from .mechanics import (
    BoundaryData,
    Gradients,
    SolidSubmesh,
    _assemble_scalar,
    _interpolate_submesh_vector_to_parent,
    create_solid_submesh,
    elastic_energy_density,
    solid_indicator_from_cells,
    solve_elasticity_solid,
    solve_regularized_scalar_form,
)
from .mmg import (
    distance_level_set_from_tags,
    run_mmg_level_set,
    run_mmg_standard,
)
from .optimization_mmg import (
    _cleanup_small_void_components,
    _num_cells,
    _num_vertices,
    _write_paraview_step,
    _write_step,
)
from .parameters import Parameters

InitialLevelSet = Callable[[np.ndarray], np.ndarray]
BoundaryMarker = Callable[[mesh.Mesh, Parameters], BoundaryData]
InitialMeshFactory = Callable[[Parameters], mesh.Mesh]


@dataclass(slots=True)
class MultiLoadState:
    """State values for the constrained bridge objective."""

    displacement: fem.Function
    compliance: float
    volume: float
    compliances: np.ndarray
    threshold: float
    worst_load: int


@dataclass(slots=True)
class MultiLoadGradients:
    """Riesz representatives of volume and load-case compliance derivatives."""

    volume: fem.Function
    compliances: list[fem.Function]


def _all_load_boundaries(
    facet_tags: mesh.MeshTags, params: Parameters, load_refs: Sequence[int]
) -> BoundaryData:
    values = np.asarray(facet_tags.values, dtype=np.int32)
    facets = np.asarray(facet_tags.indices, dtype=np.int32)
    left = facets[values == params.ref_dirichlet]
    load = facets[np.isin(values, np.asarray(load_refs, dtype=np.int32))]
    if len(left) == 0:
        raise RuntimeError("MMG mesh has no Dirichlet facets.")
    if len(load) == 0:
        raise RuntimeError("MMG mesh has no load facets.")
    return BoundaryData(left_facets=left, load_facets=load, facet_tags=facet_tags)


def _max_abs(function: fem.Function) -> float:
    local = float(np.max(np.abs(function.x.array))) if len(function.x.array) else 0.0
    return float(function.function_space.mesh.comm.allreduce(local, op=MPI.MAX))


def _linear_combination(
    space: fem.FunctionSpace, terms: list[tuple[float, fem.Function]], name: str
) -> fem.Function:
    result = fem.Function(space, name=name)
    result.x.array[:] = 0.0
    for coefficient, function in terms:
        result.x.array[:] += coefficient * function.x.array
    result.x.scatter_forward()
    return result


def _evaluate_multiload_state(
    msh: mesh.Mesh,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
    load_refs: Sequence[int],
    loads: Sequence[tuple[float, float]],
    threshold: float | None,
) -> tuple[MultiLoadState, fem.Function, SolidSubmesh, fem.Function, list[fem.Function]]:
    solid_domain = create_solid_submesh(msh, cell_tags, facet_tags, params)
    dx_solid = ufl.dx(domain=solid_domain.mesh)
    displacements: list[fem.Function] = []
    compliances: list[float] = []
    for load_ref, load in zip(load_refs, loads):
        uh = solve_elasticity_solid(
            solid_domain.mesh,
            solid_domain.boundaries,
            params,
            load_ref=load_ref,
            load=np.asarray(load, dtype=float),
        )
        displacements.append(uh)
        compliances.append(
            _assemble_scalar(
                elastic_energy_density(uh, params) * dx_solid,
                msh.comm,
            )
        )

    compliance_values = np.asarray(compliances, dtype=float)
    current_threshold = (
        0.9 * float(np.max(compliance_values)) if threshold is None else threshold
    )
    worst = int(np.argmax(compliance_values - current_threshold))
    parent_uh = _interpolate_submesh_vector_to_parent(
        displacements[worst], msh, solid_domain.cell_map
    )
    volume = _assemble_scalar(1.0 * dx_solid, msh.comm)
    solid = solid_indicator_from_cells(msh, solid_domain.parent_cells)
    state = MultiLoadState(
        displacement=parent_uh,
        compliance=float(np.max(compliance_values)),
        volume=volume,
        compliances=compliance_values,
        threshold=current_threshold,
        worst_load=worst,
    )
    return state, solid, solid_domain, displacements[worst], displacements


def _compute_multiload_gradients(
    msh: mesh.Mesh,
    boundaries: BoundaryData,
    solid_domain: SolidSubmesh,
    displacements: Sequence[fem.Function],
    params: Parameters,
) -> MultiLoadGradients:
    V = fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    ds_solid = ufl.Measure(
        "ds", domain=solid_domain.mesh, subdomain_data=solid_domain.facet_tags
    )
    entity_maps = [solid_domain.cell_map]
    grad_volume = solve_regularized_scalar_form(
        v * ds_solid(params.ref_interface),
        V,
        boundaries,
        params,
        "grad_volume",
        entity_maps=entity_maps,
    )
    grad_compliances = [
        solve_regularized_scalar_form(
            -elastic_energy_density(uh, params) * v * ds_solid(params.ref_interface),
            V,
            boundaries,
            params,
            f"grad_compliance_{i}",
            entity_maps=entity_maps,
        )
        for i, uh in enumerate(displacements)
    ]
    return MultiLoadGradients(volume=grad_volume, compliances=grad_compliances)


def _penalty_merit(state: MultiLoadState, penalty: float) -> float:
    violations = np.maximum(state.compliances - state.threshold, 0.0)
    return float(state.volume + penalty * np.dot(violations, violations))


def _penalty_speed(
    gradients: MultiLoadGradients,
    state: MultiLoadState,
    params: Parameters,
    penalty: float,
) -> fem.Function:
    violations = np.maximum(state.compliances - state.threshold, 0.0)
    terms: list[tuple[float, fem.Function]] = [(1.0, gradients.volume)]
    for violation, gradient in zip(violations, gradients.compliances):
        if violation > 0.0:
            terms.append((2.0 * penalty * float(violation), gradient))
    gradient = _linear_combination(
        gradients.volume.function_space,
        terms,
        "multiload_penalty_gradient",
    )
    scale = params.mesh_size / (params.eps**2 + _max_abs(gradient))
    return _linear_combination(
        gradients.volume.function_space,
        [(-scale, gradient)],
        "normal_speed",
    )


def _history_columns() -> list[str]:
    return [
        "iteration",
        "line_search",
        "accepted",
        "coef",
        "cells",
        "vertices",
        "volume",
        "threshold",
        "max_compliance",
        "max_violation",
        "merit",
    ]


def _write_multiload_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_history_columns())
        writer.writeheader()
        writer.writerows(rows)


def run_multiload_constraints(
    params: Parameters,
    output_dir: Path,
    *,
    initial_level_set: InitialLevelSet,
    mark_boundaries: BoundaryMarker,
    create_initial_mesh: InitialMeshFactory = create_initial_rectangle_mesh,
    load_refs: Sequence[int],
    loads: Sequence[tuple[float, float]],
    compliance_threshold_factor: float = 0.9,
    constraint_penalty: float = 2.0,
    write_xdmf: bool = True,
) -> list[dict]:
    """Run the multi-load bridge using a native penalty constrained update."""

    if len(load_refs) != len(loads):
        raise ValueError("load_refs and loads must have the same length.")

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
    phi0 = initialize_level_set(scalar_space(background.mesh), initial_level_set)
    current = run_mmg_level_set(
        background.mesh,
        phi0,
        background.facet_tags,
        background.cell_tags,
        params,
        mmg_dir,
        "initial",
    )

    threshold: float | None = None
    history: list[dict] = []
    parent_entries: list[tuple[int, Path]] = []
    solid_entries: list[tuple[int, Path]] = []
    coef = 1.0

    for iteration in range(params.max_iterations + 1):
        distance_level_set_from_tags(
            current.phi, current.cell_tags, current.facet_tags, params
        )
        boundaries = _all_load_boundaries(current.facet_tags, params, load_refs)
        state, solid, solid_domain, solid_displacement, displacements = (
            _evaluate_multiload_state(
                current.mesh,
                current.cell_tags,
                current.facet_tags,
                params,
                load_refs,
                loads,
                threshold,
            )
        )
        if threshold is None:
            threshold = compliance_threshold_factor * state.compliance
            state.threshold = threshold

        violations = state.compliances - threshold
        print(
            f"it={iteration:03d} cells={_num_cells(current):d} "
            f"V={state.volume:.6e} maxC={state.compliance:.6e} "
            f"maxH={np.max(violations):+.3e}"
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
                    parent_entries,
                    solid_entries,
                )
            break

        gradients = _compute_multiload_gradients(
            current.mesh, boundaries, solid_domain, displacements, params
        )
        speed = _penalty_speed(gradients, state, params, constraint_penalty)
        velocity = normal_velocity_from_speed(current.phi, speed, params)
        current_merit = _penalty_merit(state, constraint_penalty)
        current_tolerance = params.line_search_tolerance(iteration)

        if write_xdmf and iteration % params.output_every == 0:
            _write_step(
                output_dir / f"step_{iteration:04d}.xdmf",
                current,
                solid,
                state,
                solid_domain,
                solid_displacement,
                Gradients(compliance=speed, volume=gradients.volume),
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
                parent_entries,
                solid_entries,
                Gradients(compliance=speed, volume=gradients.volume),
                velocity,
            )

        accepted = None
        accepted_state = None
        accepted_merit = None
        accepted_k = params.max_line_search - 1

        for k in range(params.max_line_search):
            raw_trial_phi = advect_level_set(current.phi, velocity, coef, params)
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
                distance_level_set_from_tags(
                    trial.phi, trial.cell_tags, trial.facet_tags, params
                )
                trial_state, _, _, _, _ = _evaluate_multiload_state(
                    trial.mesh,
                    trial.cell_tags,
                    trial.facet_tags,
                    params,
                    load_refs,
                    loads,
                    threshold,
                )
            except RuntimeError as exc:
                print(f"  ls={k} coef={coef:.3e} failed; reject")
                if k == params.max_line_search - 1:
                    raise RuntimeError("Every constrained trial failed.") from exc
                coef = max(params.min_coef, params.line_search_decrease * coef)
                continue

            trial_merit = _penalty_merit(trial_state, constraint_penalty)
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

        if accepted is None or accepted_state is None or accepted_merit is None:
            raise RuntimeError("Line search failed without an accepted trial.")

        current = accepted
        row = {
            "iteration": iteration,
            "line_search": accepted_k,
            "accepted": True,
            "coef": coef,
            "cells": _num_cells(current),
            "vertices": _num_vertices(current),
            "volume": accepted_state.volume,
            "threshold": threshold,
            "max_compliance": accepted_state.compliance,
            "max_violation": float(np.max(accepted_state.compliances - threshold)),
            "merit": accepted_merit,
        }
        history.append(row)
        _write_multiload_history(output_dir / "history.csv", history)

    _write_multiload_history(output_dir / "history.csv", history)
    return history
