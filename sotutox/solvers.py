# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PETSc settings for sotutox's linear solves.

2D keeps the direct solver (LU): its systems are small. 3D uses Krylov solvers. The mass projections, the SUPG step
and the regularized gradients converge to a relative residual of 1e-12, which keeps the direct solver's results to
about that accuracy. Elasticity runs CG with smoothed-aggregation AMG (GAMG) given the rigid-body modes as its
near-nullspace, to the relative residual of 1e-8 that sotutox's 3D elasticity already used.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem, la
from petsc4py import PETSc

DIRECT = {"ksp_type": "preonly", "pc_type": "lu"}

ITERATIVE = {
    # consistent mass matrices: well conditioned, diagonal scaling suffices
    "mass": {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1.0e-12},
    # the SUPG step's nonsymmetric matrix: mass plus a streamline term
    "transport": {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 100,
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
        "ksp_rtol": 1.0e-12,
    },
    # (alpha^2 grad g . grad v + w g v): a mass matrix plus a scaled Laplacian
    "elliptic": {"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1.0e-12},
    # GAMG's default smoother (Chebyshev with Jacobi) suffices once it knows the rigid-body modes
    "elasticity": {"ksp_type": "cg", "ksp_rtol": 1.0e-8, "pc_type": "gamg"},
}


def options(kind: str, msh) -> dict:
    """The PETSc options for one kind of solve ("mass", "transport", "elliptic", "elasticity") on ``msh``: the
    direct solver in 2D, the iterative one in 3D."""

    if msh.topology.dim < 3:
        direct = dict(DIRECT)
        if msh.comm.size > 1:   # PETSc's own LU is serial: MUMPS on ranks
            direct["pc_factor_mat_solver_type"] = "mumps"
        return direct
    return dict(ITERATIVE[kind])


def rigid_body_modes(V: fem.FunctionSpace) -> PETSc.NullSpace:
    """The rigid-body modes of a 3D vector P1 space (three translations, three rotations), orthonormalized: the
    near-nullspace that AMG needs for elasticity."""

    index_map, bs = V.dofmap.index_map, V.dofmap.index_map_bs
    basis = [la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(6)]
    b = [v.array for v in basis]
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]
    for i in range(3):
        b[i][dofs[i]] = 1.0
    x = V.tabulate_dof_coordinates()
    blocks = V.dofmap.list.flatten()
    x0, x1, x2 = x[blocks, 0], x[blocks, 1], x[blocks, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2
    la.orthonormalize(basis)
    n = bs * index_map.size_local
    # copies: each PETSc vector keeps its own array alive, independent of the dolfinx vectors above
    vectors = [PETSc.Vec().createWithArray(np.array(v[:n]), bsize=bs, comm=V.mesh.comm) for v in b]
    return PETSc.NullSpace().create(vectors=vectors)


def solve_elasticity(problem, V: fem.FunctionSpace) -> fem.Function:
    """Solve a LinearProblem for 3D elasticity with the rigid-body modes as the AMG's near-nullspace (2D: as is)."""

    if V.mesh.topology.dim == 3:
        problem.A.setNearNullSpace(rigid_body_modes(V))
    return problem.solve()
