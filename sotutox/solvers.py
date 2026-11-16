# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PETSc settings for sotutox's linear solves.

2D keeps the direct solver (LU): its systems are small. 3D uses Krylov solvers. The mass projections, the SUPG step
and the regularized gradients converge to a relative residual of 1e-12, which keeps the direct solver's results to
about that accuracy. Elasticity runs CG with smoothed-aggregation AMG (GAMG) given the rigid-body modes as its
near-nullspace, to the relative residual of 1e-8 that sotutox's 3D elasticity already used. A 3D displacement of degree 2
or more runs the same CG to the same tolerance with a two-level p-multigrid (``PMultigrid``): GAMG on the P1 level.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem, la
from dolfinx.fem import petsc as fem_petsc
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
        return dict(DIRECT)
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


# a 3D displacement of degree 2 or more: CG with PMultigrid (True) or with GAMG on its own matrix (False, as before)
P_MULTIGRID = True


class PMultigrid:
    """Two-level p-multigrid for 3D elasticity in vector Lagrange of degree k > 1, a petsc4py python PC.

    The degree-k level is smoothed as GAMG smooths its levels (two Chebyshev steps with Jacobi). The coarse level is
    vector P1 on the same cells: its operator is P^T A P, where P interpolates P1 into the degree-k space (exactly, also
    on curved cells, where both are mapped by the same geometry), and one GAMG V-cycle solves it, given the P1 rigid-body
    modes. A is the matrix LinearProblem.solve assembles, so the coarse operator is built in ``setUp``.

    GAMG on a P2 matrix itself coarsens it about 40-fold in one step (the aggressive coarsening of its dense graph): on
    the 3D cantilever's first design (311k P2 unknowns) it takes 47 CG iterations (straight cells) and 50 (curved). This
    PC takes 20 and 22, to the same error in the energy norm; the elasticity call, assembly included, is 2.6 and 2.3
    times faster, and the compliance agrees to 1e-12 (16 Nov)."""

    def __init__(self, V: fem.FunctionSpace):
        V1 = fem.functionspace(V.mesh, ("Lagrange", 1, (V.mesh.geometry.dim,)))
        self.P = fem_petsc.interpolation_matrix(V1, V)
        self.P.assemble()
        self.modes = rigid_body_modes(V1)
        self.Ac = None
        self.mg = None

    def attach(self, ksp: PETSc.KSP) -> None:
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(self)

    def setUp(self, pc: PETSc.PC) -> None:
        A, _ = pc.getOperators()
        self.Ac = A.ptap(self.P, self.Ac)
        self.Ac.setNearNullSpace(self.modes)
        if self.mg is None:
            mg = PETSc.PC().create(A.comm)
            mg.setOptionsPrefix((pc.getOptionsPrefix() or "") + "pmg_")
            mg.setType(PETSc.PC.Type.MG)
            mg.setMGLevels(2)
            mg.setMGInterpolation(1, self.P)
            fine = mg.getMGSmoother(1)
            fine.setType(PETSc.KSP.Type.CHEBYSHEV)
            fine.setTolerances(max_it=2)
            fine.getPC().setType(PETSc.PC.Type.JACOBI)
            coarse = mg.getMGCoarseSolve()
            coarse.setType(PETSc.KSP.Type.PREONLY)
            coarse.getPC().setType(PETSc.PC.Type.GAMG)
            mg.setFromOptions()   # <prefix>pmg_mg_levels_..., <prefix>pmg_mg_coarse_... override these
            self.mg = mg
        self.mg.setOperators(A, A)
        self.mg.getMGSmoother(1).setOperators(A, A)
        self.mg.getMGCoarseSolve().setOperators(self.Ac, self.Ac)
        self.mg.setUp()

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.mg.apply(x, y)

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer) -> None:
        if self.mg is not None:
            self.mg.view(viewer)


def solve_elasticity(problem, V: fem.FunctionSpace) -> fem.Function:
    """Solve a LinearProblem for 3D elasticity with the rigid-body modes as the AMG's near-nullspace (2D: as is). A 3D
    displacement of degree 2 or more takes ``PMultigrid`` as its preconditioner (``P_MULTIGRID``), with CG and the
    tolerance unchanged."""

    if V.mesh.topology.dim == 3:
        if P_MULTIGRID and V.ufl_element().degree > 1:
            PMultigrid(V).attach(problem.solver)
        else:
            problem.A.setNearNullSpace(rigid_body_modes(V))
    return problem.solve()
