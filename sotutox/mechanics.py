# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""FEniCSx mechanics and shape-gradient solves for fitted level-set domains.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI

from . import solvers
from .parameters import Parameters


@dataclass(slots=True)
class BoundaryData:
    """Facet sets used by the elasticity and shape-gradient solves."""

    left_facets: np.ndarray
    load_facets: np.ndarray
    facet_tags: mesh.MeshTags
    roller_facets: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )


@dataclass(slots=True)
class State:
    """State solve and scalar values for one level set."""

    displacement: fem.Function
    compliance: float
    volume: float


@dataclass(slots=True)
class Gradients:
    """Regularized scalar shape gradients."""

    compliance: fem.Function
    volume: fem.Function


@dataclass(slots=True)
class SolidSubmesh:
    """Body-fitted solid mesh extracted from an MMG phase-labelled mesh."""

    mesh: mesh.Mesh
    cell_map: mesh.EntityMap
    vertex_map: mesh.EntityMap
    parent_cells: np.ndarray
    facet_tags: mesh.MeshTags
    boundaries: BoundaryData


def _cell_refs(msh: mesh.Mesh, cell_tags: mesh.MeshTags | None) -> np.ndarray:
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    refs = np.ones(num_cells, dtype=np.int32)
    if cell_tags is not None:
        refs[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
            cell_tags.values, dtype=np.int32
        )
    return refs


def _local_entities(msh: mesh.Mesh, dim: int) -> np.ndarray:
    imap = msh.topology.index_map(dim)
    return np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)


def facet_tags_from_groups(
    msh: mesh.Mesh, fdim: int, groups: list[tuple[np.ndarray, int]]
) -> mesh.MeshTags:
    groups = [(group, value) for group, value in groups if len(group) > 0]
    if not groups:
        raise ValueError("At least one facet group is required.")
    facets = np.concatenate([group for group, _ in groups]).astype(np.int32)
    values = np.concatenate(
        [np.full(len(group), value, dtype=np.int32) for group, value in groups]
    )
    order = np.argsort(facets)
    return mesh.meshtags(msh, fdim, facets[order], values[order])


def boundary_data_from_tags(facet_tags: mesh.MeshTags, params: Parameters) -> BoundaryData:
    """Build boundary data from MMG/Medit facet labels."""

    values = np.asarray(facet_tags.values, dtype=np.int32)
    indices = np.asarray(facet_tags.indices, dtype=np.int32)
    left = indices[values == params.ref_dirichlet]
    load = indices[values == params.ref_neumann]
    roller = indices[values == params.ref_dirichlet_roller]
    if len(left) == 0:
        raise RuntimeError(
            f"MMG mesh has no Dirichlet facets with reference {params.ref_dirichlet}."
        )
    if len(load) == 0:
        raise RuntimeError(
            f"MMG mesh has no Neumann/load facets with reference {params.ref_neumann}."
        )
    if params.roller_support_required and len(roller) == 0:
        raise RuntimeError(
            "MMG mesh has no roller support facets with reference "
            f"{params.ref_dirichlet_roller}."
        )
    return BoundaryData(
        left_facets=left,
        load_facets=load,
        roller_facets=roller,
        facet_tags=facet_tags,
    )


def vector_space(msh: mesh.Mesh) -> fem.FunctionSpace:
    return fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))


def epsilon(u: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    return ufl.sym(ufl.grad(u))


def sigma(u: ufl.core.expr.Expr, params: Parameters) -> ufl.core.expr.Expr:
    dim = len(u)
    return (
        2.0 * params.mu * epsilon(u)
        + params.lambda_lame * ufl.tr(epsilon(u)) * ufl.Identity(dim)
    )


def elastic_energy_density(
    u: ufl.core.expr.Expr, params: Parameters
) -> ufl.core.expr.Expr:
    """The ``A e(u):e(u)`` density used by sotuto's compliance gradient."""

    return ufl.inner(sigma(u, params), epsilon(u))


def _assemble_scalar(form: ufl.form.Form, comm: MPI.Comm) -> float:
    value = fem.assemble_scalar(fem.form(form))
    return float(comm.allreduce(value, op=MPI.SUM))


def _elasticity_bcs(V: fem.FunctionSpace, boundaries: BoundaryData) -> list[fem.DirichletBC]:
    msh = V.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    bcs: list[fem.DirichletBC] = []
    if len(boundaries.left_facets) > 0:
        zero = np.zeros(msh.geometry.dim, dtype=default_scalar_type)
        dofs = fem.locate_dofs_topological(V, fdim, boundaries.left_facets)
        bcs.append(fem.dirichletbc(zero, dofs, V))

    if len(boundaries.roller_facets) > 0:
        dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, boundaries.roller_facets)
        bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs_y, V.sub(1)))

    return bcs


def solve_elasticity_solid(
    msh: mesh.Mesh,
    boundaries: BoundaryData,
    params: Parameters,
    *,
    load_ref: int | None = None,
    load: np.ndarray | None = None,
) -> fem.Function:
    """Solve linear elasticity on the fitted solid mesh."""

    V = vector_space(msh)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=boundaries.facet_tags)

    load_values = (
        np.asarray(
            [params.load_x, params.load_y, params.load_z],
            dtype=default_scalar_type,
        )
        if load is None
        else np.asarray(load, dtype=default_scalar_type)
    )
    if len(load_values) < msh.geometry.dim:
        load_vector = np.zeros(msh.geometry.dim, dtype=default_scalar_type)
        load_vector[: len(load_values)] = load_values
    else:
        load_vector = load_values[: msh.geometry.dim]
    traction = fem.Constant(msh, load_vector)
    a = ufl.inner(sigma(u, params), epsilon(v)) * dx
    L = ufl.dot(traction, v) * ds(params.ref_neumann if load_ref is None else load_ref)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=_elasticity_bcs(V, boundaries),
        petsc_options_prefix="sotutox_solid_elasticity_",
        petsc_options=solvers.options("elasticity", msh),
    )
    uh = solvers.solve_elasticity(problem, V)
    uh.name = "solid_displacement"
    uh.x.scatter_forward()
    return uh


def create_solid_submesh(
    msh: mesh.Mesh,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
) -> SolidSubmesh:
    """Create the solid submesh ``reg(x, y) == REFINT`` used by sotuto."""

    tdim = msh.topology.dim
    refs = _cell_refs(msh, cell_tags)
    solid_cells = np.flatnonzero(refs == params.ref_solid).astype(np.int32)
    if len(solid_cells) == 0:
        raise RuntimeError("MMG mesh has no solid cells with REFINT/ref_solid.")

    def build(cells: np.ndarray) -> SolidSubmesh:
        solid_mesh, cell_map, vertex_map, _ = mesh.create_submesh(
            msh, tdim, cells.astype(np.int32)
        )
        solid_facet_tags = mesh.transfer_meshtags_to_submesh(
            facet_tags, solid_mesh, vertex_map, cell_map
        )
        boundaries = boundary_data_from_tags(solid_facet_tags, params)
        return SolidSubmesh(
            mesh=solid_mesh,
            cell_map=cell_map,
            vertex_map=vertex_map,
            parent_cells=np.asarray(cells, dtype=np.int32),
            facet_tags=solid_facet_tags,
            boundaries=boundaries,
        )

    return build(solid_cells)


def solid_indicator_from_cells(
    msh: mesh.Mesh, solid_cells: np.ndarray
) -> fem.Function:
    """Return a DG0 solid indicator from selected parent cells."""

    Q = fem.functionspace(msh, ("DG", 0))
    solid_mask = np.zeros(msh.topology.index_map(msh.topology.dim).size_local, dtype=bool)
    solid_mask[np.asarray(solid_cells, dtype=np.int32)] = True

    solid = fem.Function(Q, name="solid_indicator")
    for cell, is_solid in enumerate(solid_mask):
        dofs = Q.dofmap.cell_dofs(cell)
        solid.x.array[dofs] = 1.0 if is_solid else 0.0
    solid.x.scatter_forward()
    return solid


def _interpolate_submesh_vector_to_parent(
    source: fem.Function, target_mesh: mesh.Mesh, cell_map: mesh.EntityMap
) -> fem.Function:
    """Interpolate a fitted-submesh vector P1 function onto its parent mesh."""

    target = fem.Function(vector_space(target_mesh), name="displacement")
    target.x.array[:] = 0.0
    tdim = source.function_space.mesh.topology.dim
    sub_cells = _local_entities(source.function_space.mesh, tdim)
    parent_cells = cell_map.sub_topology_to_topology(sub_cells, inverse=False)
    target.interpolate(source, cells0=sub_cells, cells1=parent_cells)
    target.x.scatter_forward()
    return target


def evaluate_state_fitted(
    msh: mesh.Mesh,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
) -> tuple[State, fem.Function, SolidSubmesh, fem.Function]:
    """Evaluate the faithful sotuto state on the body-fitted solid mesh."""

    solid_domain = create_solid_submesh(msh, cell_tags, facet_tags, params)
    solid_uh = solve_elasticity_solid(
        solid_domain.mesh, solid_domain.boundaries, params
    )
    parent_uh = _interpolate_submesh_vector_to_parent(
        solid_uh, msh, solid_domain.cell_map
    )

    dx_solid = ufl.dx(domain=solid_domain.mesh)
    compliance = _assemble_scalar(
        elastic_energy_density(solid_uh, params) * dx_solid, msh.comm
    )
    volume = _assemble_scalar(1.0 * dx_solid, msh.comm)
    solid = solid_indicator_from_cells(msh, solid_domain.parent_cells)
    return (
        State(displacement=parent_uh, compliance=compliance, volume=volume),
        solid,
        solid_domain,
        solid_uh,
    )


def _zero_on_load_bc(V: fem.FunctionSpace, boundaries: BoundaryData) -> fem.DirichletBC:
    V.mesh.topology.create_connectivity(V.mesh.topology.dim - 1, V.mesh.topology.dim)
    dofs = fem.locate_dofs_topological(
        V, V.mesh.topology.dim - 1, boundaries.load_facets
    )
    return fem.dirichletbc(default_scalar_type(0.0), dofs, V)


def solve_regularized_scalar_form(
    rhs_form: ufl.form.Form,
    V: fem.FunctionSpace,
    boundaries: BoundaryData,
    params: Parameters,
    name: str,
    entity_maps: list[mesh.EntityMap] | None = None,
    mass_weight: ufl.core.expr.Expr | float = 1.0,
) -> fem.Function:
    """Solve ``(alpha^2 grad g, grad v) + (g, v) = rhs`` for a linear form."""

    g = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=V.mesh)
    a = (
        params.regularization**2 * ufl.dot(ufl.grad(g), ufl.grad(v))
        + mass_weight * g * v
    ) * dx
    problem = petsc.LinearProblem(
        a,
        rhs_form,
        bcs=[_zero_on_load_bc(V, boundaries)],
        petsc_options_prefix=f"sotutox_{name}_",
        petsc_options=solvers.options("elliptic", V.mesh),
        entity_maps=entity_maps,
    )
    result = problem.solve()
    result.name = name
    result.x.scatter_forward()
    return result


def compute_shape_gradients_fitted(
    msh: mesh.Mesh,
    boundaries: BoundaryData,
    solid_domain: SolidSubmesh,
    solid_displacement: fem.Function,
    params: Parameters,
) -> Gradients:
    """Compute sotuto's interface shape gradients on the fitted solid boundary."""

    V = fem.functionspace(msh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    ds_solid = ufl.Measure(
        "ds", domain=solid_domain.mesh, subdomain_data=solid_domain.facet_tags
    )
    energy = elastic_energy_density(solid_displacement, params)
    entity_maps = [solid_domain.cell_map]
    grad_compliance = solve_regularized_scalar_form(
        -energy * v * ds_solid(params.ref_interface),
        V,
        boundaries,
        params,
        "grad_compliance",
        entity_maps=entity_maps,
    )
    grad_volume = solve_regularized_scalar_form(
        v * ds_solid(params.ref_interface),
        V,
        boundaries,
        params,
        "grad_volume",
        entity_maps=entity_maps,
    )
    return Gradients(compliance=grad_compliance, volume=grad_volume)


def regularized_inner(
    f: fem.Function,
    g: fem.Function,
    params: Parameters,
    mass_weight: ufl.core.expr.Expr | float = 1.0,
) -> float:
    """H1 inner product used by sotuto's descent script."""

    dx = ufl.dx(domain=f.function_space.mesh)
    form = (
        params.regularization**2 * ufl.dot(ufl.grad(f), ufl.grad(g))
        + mass_weight * f * g
    ) * dx
    return _assemble_scalar(form, f.function_space.mesh.comm)
