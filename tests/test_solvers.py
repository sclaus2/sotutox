# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotutox import solvers
from sotutox.redistance import graph_signed_distance


def _cube() -> mesh.Mesh:
    return mesh.create_box(MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)), (8, 4, 4))


def test_options_are_direct_in_2d_and_iterative_in_3d() -> None:
    square = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)

    assert solvers.options("mass", square) == solvers.DIRECT
    assert solvers.options("elasticity", square) == solvers.DIRECT
    assert solvers.options("mass", _cube())["ksp_type"] == "cg"
    assert solvers.options("transport", _cube())["ksp_type"] == "gmres"


def test_3d_mass_projection_matches_the_direct_solve() -> None:
    msh = _cube()
    W = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    x = ufl.SpatialCoordinate(msh)
    u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    target = ufl.as_vector((ufl.sin(3 * x[0]), x[1] * x[2], ufl.cos(x[0] + x[1])))
    results = []
    for options in (solvers.DIRECT, solvers.options("mass", msh)):
        problem = petsc.LinearProblem(ufl.inner(u, v) * ufl.dx, ufl.inner(target, v) * ufl.dx,
                                      petsc_options_prefix=f"test_mass_{len(results)}_", petsc_options=options)
        results.append(problem.solve().x.array.copy())

    np.testing.assert_allclose(results[1], results[0], rtol=0.0, atol=1.0e-10)


def test_3d_elasticity_with_rigid_body_modes_matches_the_direct_solve() -> None:
    msh = _cube()
    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    mu, lmbda = 1.0, 1.25

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(3)

    clamped = mesh.locate_entities_boundary(msh, 2, lambda p: np.isclose(p[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3), fem.locate_dofs_topological(V, 2, clamped), V)
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(fem.Constant(msh, np.array([0.0, -1.0, 0.3])), v) * ufl.dx
    direct = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="test_elasticity_lu_",
                                 petsc_options=solvers.DIRECT).solve()
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="test_elasticity_amg_",
                                  petsc_options=solvers.options("elasticity", msh))
    amg = solvers.solve_elasticity(problem, V)

    assert problem.solver.getConvergedReason() > 0
    scale = np.abs(direct.x.array).max()
    np.testing.assert_allclose(amg.x.array, direct.x.array, rtol=0.0, atol=1.0e-6 * scale)


def test_3d_p2_elasticity_with_p_multigrid_matches_the_direct_solve_and_gamg(monkeypatch) -> None:
    msh = _cube()
    V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    mu, lmbda = 1.0, 1.25

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(3)

    clamped = mesh.locate_entities_boundary(msh, 2, lambda p: np.isclose(p[0], 0.0))
    bc = fem.dirichletbc(np.zeros(3), fem.locate_dofs_topological(V, 2, clamped), V)
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(fem.Constant(msh, np.array([0.0, -1.0, 0.3])), v) * ufl.dx
    direct = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix="test_p2_lu_",
                                 petsc_options=solvers.DIRECT).solve()
    scale = np.abs(direct.x.array).max()
    iterations = {}
    for p_multigrid in (True, False):
        monkeypatch.setattr(solvers, "P_MULTIGRID", p_multigrid)
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"test_p2_{p_multigrid}_",
                                      petsc_options=solvers.options("elasticity", msh))
        uh = solvers.solve_elasticity(problem, V)
        assert problem.solver.getPC().getType() == ("python" if p_multigrid else "gamg")
        assert problem.solver.getConvergedReason() > 0
        iterations[p_multigrid] = problem.solver.getIterationNumber()
        np.testing.assert_allclose(uh.x.array, direct.x.array, rtol=0.0, atol=1.0e-6 * scale)

    assert iterations[True] < iterations[False]


def test_graph_signed_distance_propagates_from_the_nearest_seed() -> None:
    # crossings at x = 0.5 and x = 3.25 on a path 0-1-2-3-4: the middle vertex is nearer to the right one
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    values = np.array([-0.5, 0.5, 1.5, 0.05, -0.15])

    result = graph_signed_distance(points, edges, values)

    np.testing.assert_allclose(result, np.array([-0.5, 0.5, 1.25, 0.25, -0.75]))
