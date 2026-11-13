# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Level-set advection helpers for fitted level-set optimization.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
from scipy.spatial import cKDTree

from . import solvers
from .parameters import Parameters


def _points_3d(points: np.ndarray, gdim: int) -> np.ndarray:
    """Return points with the 3-coordinate shape expected by DOLFINx eval."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    out = np.zeros((points.shape[0], 3), dtype=np.float64)
    out[:, :gdim] = points[:, :gdim]
    return out


def _locate_cells(msh: mesh.Mesh, points: np.ndarray) -> np.ndarray:
    """Locate a containing cell for each point, falling back to nearest cells."""

    tdim = msh.topology.dim
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    tree = geometry.bb_tree(msh, tdim, padding=1.0e-12)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(msh, candidates, points)
    for i in range(points.shape[0]):
        links = collisions.links(i)
        if len(links) > 0:
            cells[i] = links[0]

    missing = np.flatnonzero(cells < 0)
    if missing.size:
        num_cells = msh.topology.index_map(tdim).size_local
        local_cells = np.arange(num_cells, dtype=np.int32)
        midpoint_tree = geometry.create_midpoint_tree(msh, tdim, local_cells)
        cells[missing] = geometry.compute_closest_entity(
            tree, midpoint_tree, msh, points[missing]
        )
    return cells


def evaluate_function_at_points(function: fem.Function, points: np.ndarray) -> np.ndarray:
    """Evaluate a scalar or vector function at physical points."""

    msh = function.function_space.mesh
    points3 = _points_3d(points, msh.geometry.dim)
    cells = _locate_cells(msh, points3)
    return np.asarray(function.eval(points3, cells), dtype=float)


def _scalar_values_at_space_dofs(
    function: fem.Function, target_space: fem.FunctionSpace
) -> np.ndarray:
    """Return scalar P1 values at the block dof coordinates of ``target_space``."""

    source_space = function.function_space
    if source_space.dofmap.bs != 1:
        raise RuntimeError("Expected a scalar P1 function.")
    gdim = source_space.mesh.geometry.dim
    source_coords = source_space.tabulate_dof_coordinates()[:, :gdim]
    target_coords = target_space.tabulate_dof_coordinates()[:, :gdim]
    distances, dofs = cKDTree(source_coords).query(target_coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match scalar speed dofs to vector dofs.")
    return np.asarray(function.x.array.reshape((-1, 1))[dofs, 0], dtype=np.float64)


def normal_velocity_from_speed(
    phi: fem.Function, speed: fem.Function, params: Parameters
) -> fem.Function:
    """Build the FreeFEM-style velocity ``speed * n`` on vector P1 dofs."""

    msh = phi.function_space.mesh
    gdim = msh.geometry.dim
    W = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    normal_trial = ufl.TrialFunction(W)
    test = ufl.TestFunction(W)
    raw_normal = ufl.grad(phi) / ufl.sqrt(
        params.eps + ufl.inner(ufl.grad(phi), ufl.grad(phi))
    )
    dx = ufl.dx(domain=msh)
    problem = petsc.LinearProblem(
        ufl.inner(normal_trial, test) * dx,
        ufl.inner(raw_normal, test) * dx,
        petsc_options_prefix="sotutox_normal_velocity_",
        petsc_options=solvers.options("mass", msh),
    )
    projected_normal = problem.solve()
    normal_values = projected_normal.x.array.reshape((-1, W.dofmap.bs))[:, :gdim]
    norms = np.sqrt(params.eps**2 + np.sum(normal_values**2, axis=1))
    normal_values[:] /= norms[:, None]

    velocity = fem.Function(W, name="descent_velocity")
    speed_values = _scalar_values_at_space_dofs(speed, W)
    velocity_values = velocity.x.array.reshape((-1, W.dofmap.bs))[:, :gdim]
    velocity_values[:] = speed_values[:, None] * normal_values
    velocity.x.scatter_forward()
    return velocity


def advect_characteristics(
    phi: fem.Function,
    velocity: fem.Function,
    step: float,
    *,
    method: str = "midpoint",
) -> fem.Function:
    """Advect ``phi`` with ``dphi/dt + velocity · grad(phi) = 0``.

    The method traces backward characteristics with P1 velocity interpolation
    and samples the old scalar field at the departure point using native
    DOLFINx point location.
    """

    if phi.function_space.mesh.comm.size != 1:
        raise RuntimeError("sotutox characteristic advection is serial-only.")

    msh = phi.function_space.mesh
    gdim = msh.geometry.dim
    dof_points = phi.function_space.tabulate_dof_coordinates()[:, :gdim]

    if method == "midpoint":
        v0 = evaluate_function_at_points(velocity, dof_points)[:, :gdim]
        half_points = dof_points - 0.5 * step * v0
        v_mid = evaluate_function_at_points(velocity, half_points)[:, :gdim]
        departure_points = dof_points - step * v_mid
    elif method == "euler":
        v0 = evaluate_function_at_points(velocity, dof_points)[:, :gdim]
        departure_points = dof_points - step * v0
    else:
        raise ValueError(f"Unknown advection method {method!r}.")

    values = evaluate_function_at_points(phi, departure_points).reshape(-1)
    advected = fem.Function(phi.function_space, name="phi")
    advected.x.array[:] = values
    advected.x.scatter_forward()
    return advected


class SUPGAdvector:
    """Reusable SUPG advection problem for one scalar/vector space pair."""

    def __init__(
        self,
        phi_space: fem.FunctionSpace,
        velocity_space: fem.FunctionSpace,
        params: Parameters,
    ) -> None:
        self.phi_space = phi_space
        self.velocity_space = velocity_space
        msh = phi_space.mesh

        self.phi_old = fem.Function(phi_space, name="phi_old")
        self.velocity = fem.Function(velocity_space, name="supg_velocity")
        self.result = fem.Function(phi_space, name="phi")
        self.dt = fem.Constant(msh, 1.0)
        self.tau_scale = fem.Constant(msh, params.supg_tau_scale)
        self.eps = fem.Constant(msh, params.eps)

        trial = ufl.TrialFunction(phi_space)
        test = ufl.TestFunction(phi_space)
        dx = ufl.dx(domain=msh)

        beta = self.velocity
        h = ufl.CellDiameter(msh)
        beta_norm = ufl.sqrt(ufl.inner(beta, beta) + self.eps**2)
        tau = self.tau_scale / ufl.sqrt(
            (2.0 / self.dt) ** 2 + (2.0 * beta_norm / h) ** 2
        )
        streamline_test = ufl.dot(beta, ufl.grad(test))
        residual_trial = trial / self.dt + ufl.dot(beta, ufl.grad(trial))
        residual_old = self.phi_old / self.dt

        a = (
            (trial / self.dt) * test
            + ufl.dot(beta, ufl.grad(trial)) * test
            + tau * streamline_test * residual_trial
        ) * dx
        L = (residual_old * test + tau * streamline_test * residual_old) * dx

        self.problem = petsc.LinearProblem(
            a,
            L,
            u=self.result,
            petsc_options_prefix=f"sotutox_supg_advection_{id(self):x}_",
            petsc_options=solvers.options("transport", msh),
        )

    def solve(
        self,
        phi: fem.Function,
        velocity: fem.Function,
        step: float,
        params: Parameters,
    ) -> fem.Function:
        """Update coefficients, assemble with cached forms, and solve."""

        self.dt.value = float(step)
        self.tau_scale.value = float(params.supg_tau_scale)
        self.eps.value = float(params.eps)
        self.phi_old.x.array[:] = phi.x.array
        self.phi_old.x.scatter_forward()
        self.velocity.x.array[:] = velocity.x.array
        self.velocity.x.scatter_forward()

        self.problem.solve()
        copied = fem.Function(self.phi_space, name="phi")
        copied.x.array[:] = self.result.x.array
        copied.x.scatter_forward()
        return copied


_MAX_SUPG_CACHE_SIZE = 4
_SUPG_ADVECTION_CACHE: OrderedDict[tuple[int, int], SUPGAdvector] = OrderedDict()


def advect_supg(
    phi: fem.Function,
    velocity: fem.Function,
    step: float,
    params: Parameters,
) -> fem.Function:
    """Advect one implicit Euler step with cached SUPG forms."""

    if step <= 0.0:
        result = fem.Function(phi.function_space, name="phi")
        result.x.array[:] = phi.x.array
        result.x.scatter_forward()
        return result

    key = (id(phi.function_space), id(velocity.function_space))
    advector = _SUPG_ADVECTION_CACHE.get(key)
    if advector is None:
        advector = SUPGAdvector(phi.function_space, velocity.function_space, params)
        _SUPG_ADVECTION_CACHE[key] = advector
        while len(_SUPG_ADVECTION_CACHE) > _MAX_SUPG_CACHE_SIZE:
            _SUPG_ADVECTION_CACHE.popitem(last=False)
    else:
        _SUPG_ADVECTION_CACHE.move_to_end(key)
    return advector.solve(phi, velocity, step, params)


def advect_level_set(
    phi: fem.Function,
    velocity: fem.Function,
    step: float,
    params: Parameters,
) -> fem.Function:
    """Dispatch to the selected native FEniCSx advection backend."""

    if params.advection_backend == "supg":
        return advect_supg(phi, velocity, step, params)
    if params.advection_backend in {"midpoint", "euler"}:
        return advect_characteristics(
            phi,
            velocity,
            step,
            method=params.advection_backend,
        )
    raise ValueError(f"Unknown advection backend {params.advection_backend!r}.")
