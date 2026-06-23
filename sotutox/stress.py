# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stress objective support for the ``sotuto/stress`` L-beam example."""

from __future__ import annotations

from collections.abc import Callable

import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc

from .mechanics import (
    BoundaryData,
    Gradients,
    SolidSubmesh,
    State,
    _assemble_scalar,
    _elasticity_bcs,
    _interpolate_submesh_vector_to_parent,
    create_solid_submesh,
    epsilon,
    regularized_inner,
    sigma,
    solid_indicator_from_cells,
    solve_elasticity_solid,
    solve_regularized_scalar_form,
    vector_space,
)
from .optimization import DescentData, compute_descent_with_inner
from .parameters import Parameters

WeightFactory = Callable[[mesh.Mesh], ufl.core.expr.Expr]


def stress_density(u: ufl.core.expr.Expr, params: Parameters) -> ufl.core.expr.Expr:
    """Return ``sigma(u):sigma(u)`` used by ``sotuto/stress``."""

    return ufl.inner(sigma(u, params), sigma(u, params))


def solve_stress_adjoint(
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    params: Parameters,
    stress_weight: WeightFactory,
) -> fem.Function:
    """Solve the adjoint equation from ``sotuto/stress/sources/adjoint.edp``."""

    msh = solid_domain.mesh
    V = vector_space(msh)
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=msh)

    a = ufl.inner(sigma(p, params), epsilon(v)) * dx
    rhs_density = -2.0 * stress_weight(msh) * ufl.inner(
        sigma(solid_displacement, params), sigma(v, params)
    )
    L = rhs_density * dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=_elasticity_bcs(V, solid_domain.boundaries),
        petsc_options_prefix="sotutox_stress_adjoint_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    adjoint = problem.solve()
    adjoint.name = "solid_stress_adjoint"
    adjoint.x.scatter_forward()
    return adjoint


def evaluate_stress_state_fitted(
    msh: mesh.Mesh,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
    stress_weight: WeightFactory,
) -> tuple[State, fem.Function, SolidSubmesh, fem.Function]:
    """Evaluate stress objective and volume on the body-fitted solid mesh."""

    solid_domain = create_solid_submesh(msh, cell_tags, facet_tags, params)
    solid_uh = solve_elasticity_solid(
        solid_domain.mesh, solid_domain.boundaries, params
    )
    parent_uh = _interpolate_submesh_vector_to_parent(
        solid_uh, msh, solid_domain.cell_map
    )

    dx_solid = ufl.dx(domain=solid_domain.mesh)
    stress = _assemble_scalar(
        stress_weight(solid_domain.mesh)
        * stress_density(solid_uh, params)
        * dx_solid,
        msh.comm,
    )
    volume = _assemble_scalar(1.0 * dx_solid, msh.comm)
    solid = solid_indicator_from_cells(msh, solid_domain.parent_cells)
    return (
        State(displacement=parent_uh, compliance=stress, volume=volume),
        solid,
        solid_domain,
        solid_uh,
    )


def compute_stress_shape_gradients_fitted(
    msh: mesh.Mesh,
    boundaries: BoundaryData,
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    params: Parameters,
    stress_weight: WeightFactory,
    regularization_mass: WeightFactory,
) -> Gradients:
    """Compute stress and volume shape gradients for the L-beam example."""

    V = fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    ds_solid = ufl.Measure(
        "ds", domain=solid_domain.mesh, subdomain_data=solid_domain.facet_tags
    )
    adjoint = solve_stress_adjoint(
        solid_domain, solid_displacement, params, stress_weight
    )
    weight = stress_weight(solid_domain.mesh)
    density = stress_density(solid_displacement, params)
    coupling = ufl.inner(sigma(solid_displacement, params), epsilon(adjoint))
    entity_maps = [solid_domain.cell_map]
    mass_weight = regularization_mass(msh)

    grad_stress = solve_regularized_scalar_form(
        (weight * density + coupling) * v * ds_solid(params.ref_interface),
        V,
        boundaries,
        params,
        "grad_stress",
        entity_maps=entity_maps,
        mass_weight=mass_weight,
    )
    grad_volume = solve_regularized_scalar_form(
        v * ds_solid(params.ref_interface),
        V,
        boundaries,
        params,
        "grad_volume",
        entity_maps=entity_maps,
        mass_weight=mass_weight,
    )
    return Gradients(compliance=grad_stress, volume=grad_volume)


def compute_stress_descent(
    gradients: Gradients,
    state: State,
    params: Parameters,
    max_norm_xi_objective: float,
    regularization_mass: WeightFactory,
) -> DescentData:
    """Use the stress example's weighted ``psreg`` inner product."""

    mass_weight = regularization_mass(gradients.compliance.function_space.mesh)
    return compute_descent_with_inner(
        gradients,
        state,
        params,
        max_norm_xi_objective,
        lambda f, g: regularized_inner(f, g, params, mass_weight=mass_weight),
    )
