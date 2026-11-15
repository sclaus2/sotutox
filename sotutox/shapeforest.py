# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ShapeForest's mesher as the optimization loop's remesher, instead of mmg (``run(..., remesher=ShapeForestRemesher(...))``).

Each level-set call hands the mesher phi on the mesh it lives on: its P1 values, or with ``representation="p1g"`` the values
and G, the lumped L2 projection of their gradient (a Hermite level set: the remesh corrects its interface onto that
curved zero set, as mmg places its points on Bezier patches built from averaged normals). ``sm.planar.conform`` (2D)
or ``sm.volume.conform`` (3D) cuts and remeshes it: the interface on the zero set within ``tol``, parts below the
resolution joined, solid parts that do not reach the ``anchors`` patch removed (as mmg's -rmc), the load and clamp
``patches`` kept as tagged facets. ``no_cross_edges`` keeps vertices inside every wall and bar thicker than the floor
(``cross_floor`` x ``h_min``): the level set this loop rebuilds from the tags is 0 at every interface vertex, so a
wall with none inside has no sign and opens at the next step (mmg's FEM mode does the same). With P1+G, changed_only and
no_cross_edges the 3D cantilever at h 0.07 reached mmg3d's design (J 0.170 at iteration 60).

On MPI ranks the loop starts from its initial mesh (``start="initial"``). In 2D each rank conforms its own cells (P1); in
3D too for P1 (ShapeForest's M2b: the cut on each rank's own cells, the remesh from the parts), and for P1+G with
``parts=True`` (G assembled on the ranks); otherwise every rank is given the whole mesh and phi, and with changed_only
the remesh's rounds start around the change.

ShapeForest is optional: importing this module raises ImportError when it is not installed. Its compiled mesher
(``shapeforest._sf_mesher``, branch ``meshing/pass-rework`` or later; 3D on MPI ranks ``meshing/volume-ranks`` from M2b,
ABI 57, an MPI build) must match its Python package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

import shapeforest  # noqa: F401  (ImportError here: ShapeForest is not installed)
from shapeforest import mesh as sm

from .mmg import (
    RemeshedLevelSet,
    RemeshedMesh,
    _phase_seed_values,
    _set_vertex_values,
    _vertex_values,
    run_mmg_standard,
)
from .parameters import Parameters

_P1 = {2: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
       3: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])}


def cells_of(msh: mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The mesh's points (n, d) and cells (m, d + 1), positively oriented."""
    d = msh.topology.dim
    X = np.ascontiguousarray(msh.geometry.x[:, :d], np.float64)
    T = np.array(msh.geometry.dofmaps[0], np.int64)
    p = X[T]
    J = np.stack([p[:, j + 1] - p[:, 0] for j in range(d)], axis=2)
    flip = np.linalg.det(J) < 0.0
    T[flip] = T[flip][:, [1, 0] + list(range(2, d + 1))]
    return X, np.ascontiguousarray(T)


def lumped_gradient(X: np.ndarray, T: np.ndarray, f: np.ndarray) -> np.ndarray:
    """At the vertices: the lumped L2 projection of the P1 gradient, the cells' gradients weighted by their measure."""
    T = np.asarray(T, np.int64)
    d = X.shape[1]
    J = np.stack([X[T[:, j + 1]] - X[T[:, 0]] for j in range(d)], axis=2)   # columns x_j - x_0
    df = np.stack([f[T[:, j + 1]] - f[T[:, 0]] for j in range(d)], axis=1)
    g = np.linalg.solve(np.transpose(J, (0, 2, 1)), df[:, :, None])[:, :, 0]
    w = np.abs(np.linalg.det(J))
    num, den = np.zeros((len(X), d)), np.zeros(len(X))
    for j in range(d + 1):
        np.add.at(num, T[:, j], w[:, None] * g)
        np.add.at(den, T[:, j], w)
    return np.ascontiguousarray(num / den[:, None])


def lumped_gradient_fem(phi) -> fem.Function:
    """:func:`lumped_gradient` on phi's mesh on any number of ranks: the lumped L2 projection of grad(phi) as a P1 vector
    function, assembled (per vertex the cells' gradients weighted by their measure), its ghost values up to date."""
    from dolfinx import la
    msh = phi.function_space.mesh
    d = msh.topology.dim
    W = fem.functionspace(msh, ("Lagrange", 1, (d,)))
    S = fem.functionspace(msh, ("Lagrange", 1))
    num = fem.assemble_vector(fem.form(ufl.inner(ufl.grad(phi), ufl.TestFunction(W)) * ufl.dx))
    den = fem.assemble_vector(fem.form(ufl.TestFunction(S) * ufl.dx))
    num.scatter_reverse(la.InsertMode.add)
    den.scatter_reverse(la.InsertMode.add)
    g = fem.Function(W, name="G")
    n = W.dofmap.index_map.size_local
    g.x.array[:n * d] = num.array[:n * d] / np.repeat(den.array[:n], d)
    g.x.scatter_forward()
    return g


def to_dolfinx(out, params: Parameters, patch_refs: list[int], d: int) -> tuple[mesh.Mesh, mesh.MeshTags, mesh.MeshTags]:
    """A conform result (dimension d) as a DOLFINx mesh with sotutox's references: material 0 (phi < 0) solid, else void;
    the outer facets of patch k ``patch_refs[k]`` (the others 0); the interface ``params.ref_interface``."""
    X = np.asarray(out.points, np.float64)
    T = np.asarray(out.tets if d == 3 else out.tris, np.int64)
    cell = "tetrahedron" if d == 3 else "triangle"
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell, 1, shape=(d,), dtype=np.float64))
    msh = mesh.create_mesh(MPI.COMM_WORLD, T, domain, X)
    refs = np.where(np.asarray(out.materials) == 0, params.ref_solid, params.ref_void).astype(np.int32)
    n_cells = msh.topology.index_map(d).size_local
    cell_tags = mesh.meshtags(msh, d, np.arange(n_cells, dtype=np.int32), refs[np.asarray(msh.topology.original_cell_index)])
    msh.topology.create_connectivity(d - 1, 0)
    msh.topology.create_connectivity(d - 1, d)
    node_of_input = np.empty(len(X), np.int64)
    node_of_input[np.asarray(msh.geometry.input_global_indices, np.int64)] = np.arange(len(X))
    n_facets = msh.topology.index_map(d - 1).size_local
    ends = np.sort(mesh.entities_to_geometry(msh, d - 1, np.arange(n_facets, dtype=np.int32)).astype(np.int64), axis=1)
    n = np.int64(len(X))
    code = ends[:, 0] * n + ends[:, 1] if d == 2 else (ends[:, 0] * n + ends[:, 1]) * n + ends[:, 2]
    order = np.argsort(code)

    def facets_of(F):
        f = np.sort(node_of_input[np.asarray(F, np.int64)], axis=1)
        c = f[:, 0] * n + f[:, 1] if d == 2 else (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
        return order[np.searchsorted(code[order], c)]

    outer_facets = out.outer_faces if d == 3 else out.outer_edges
    outer_patch = out.outer_face_patch if d == 3 else out.outer_edge_patch
    outer = facets_of(outer_facets)
    values = np.zeros(len(outer), np.int32)
    if outer_patch is not None:
        patch = np.asarray(outer_patch)
        for k, ref in enumerate(patch_refs):
            values[patch == k] = ref
    inner = facets_of(out.faces if d == 3 else out.edges)
    indices = np.concatenate([outer, inner]).astype(np.int32)
    values = np.concatenate([values, np.full(len(inner), params.ref_interface, np.int32)])
    rows = np.argsort(indices)
    return msh, cell_tags, mesh.meshtags(msh, d - 1, indices[rows], values[rows])


@dataclass
class ShapeForestRemesher:
    """The loop's two remesh calls by ShapeForest (see the module's docstring).

    ``patches``: (Shape, reference) pairs, the outer facets in each Shape tagged with its reference (the load
    ``params.ref_neumann``, the clamp ``params.ref_dirichlet``). ``anchors``: conform's, e.g. ``{0: "clamp"}`` -- solid parts
    that do not reach the patch named "clamp" are removed. ``h``, ``tol``, ``h_min``, ``min_part``, ``min_radius``,
    ``snap``: conform's. ``representation`` of phi: ``"p1"`` or ``"p1g"``. ``changed_only``: remesh only around the change.
    ``no_cross_edges`` and ``cross_floor`` (x h_min; 3D only, 2D's floor is 2 h_min): the rule above. ``start``: the first mesh, which phi0 is sampled on --
    ``"mmg"`` (sotutox's mmg remesh of the initial mesh, as the mmg loop starts) or ``"initial"`` (the initial mesh as it is:
    no mmg anywhere). ``rounds``: on MPI ranks in 3D, the remesh's rounds (``volume.conform(rounds=)``: each round moves
    the parts' borders). ``parts``: on MPI ranks in 3D without changed_only, the cut on each rank's own cells
    (ShapeForest's M2b; True) or the whole input on every rank (False); None: P1 on the parts, P1+G whole (on 2 ranks
    the P1+G cut on the parts took 1.8 times the whole cut's time, the call 1.1 times, 15 November). ``options``:
    conform's ``options``. ``failures``: a folder that keeps the input of every failed call.

    On MPI ranks (``msh.comm.size > 1``): ``start="initial"``. In 2D each rank conforms its own cells (P1 only); in 3D
    every rank is given the whole input and the remesh runs on the ranks, with P1 or P1+G and changed_only as on one
    process (ShapeForest's M4: the rounds start around the change)."""

    patches: list = field(default_factory=list)
    h: float = 0.07
    tol: float = 0.0175
    h_min: float | None = None
    anchors: dict | None = None
    min_part: float = 0.0
    min_radius: object = "resolution"
    snap: float = 0.1
    representation: str = "p1g"
    changed_only: bool = True
    no_cross_edges: bool = True
    cross_floor: float | None = 0.5
    start: str = "mmg"
    rounds: int = 4
    parts: bool | None = None
    options: dict = field(default_factory=dict)
    failures: Path | None = None
    calls: list = field(default_factory=list)   # per call: name, seconds, cells, ok

    def __post_init__(self):
        if self.representation not in ("p1", "p1g"):
            raise ValueError(f"ShapeForestRemesher: representation must be 'p1' or 'p1g', got {self.representation!r}")
        if self.start not in ("mmg", "initial"):
            raise ValueError(f"ShapeForestRemesher: start must be 'mmg' or 'initial', got {self.start!r}")

    def standard(self, msh, facet_tags, cell_tags, params: Parameters, work_dir, name) -> RemeshedMesh:
        """The first mesh, which phi0 is sampled on."""
        if self.start == "mmg":
            if msh.comm.size > 1:
                raise ValueError("ShapeForestRemesher on MPI ranks: start='mmg' runs mmg, which is serial; valid: "
                                 "start='initial'")
            return run_mmg_standard(msh, facet_tags, cell_tags, params, work_dir, name)
        d = msh.topology.dim
        n_cells = msh.topology.index_map(d).size_local
        tags = mesh.meshtags(msh, d, np.arange(n_cells, dtype=np.int32), np.full(n_cells, params.ref_solid, np.int32))
        return RemeshedMesh(msh, tags, facet_tags)

    def source(self, X: np.ndarray, T: np.ndarray, values: np.ndarray):
        d = X.shape[1]
        T32 = np.ascontiguousarray(T, np.int32)
        if self.representation == "p1g":
            return sm.sources.lagrange(points=X, cells=T32, values=values, dofmap=T32, nodes=_P1[d], dtype="float64",
                                       gradient=lumped_gradient(X, T, values), between="hermite")
        return sm.sources.lagrange(points=X, cells=T32, values=values, dofmap=T32, nodes=_P1[d], dtype="float64")

    def _kw(self) -> dict:
        kw = dict(h=self.h, tol=self.tol, h_min=self.h_min, patches=[s for s, _ in self.patches], anchors=self.anchors,
                  min_part=self.min_part, backend="cpp", changed_only=self.changed_only,
                  no_cross_edges=self.no_cross_edges, snap=self.snap)
        if self.min_radius is not None:
            kw["min_radius"] = self.min_radius
        return kw

    def conform(self, X: np.ndarray, T: np.ndarray, values: np.ndarray, comm=None):
        """conform on (X, T) with phi's vertex values: the result and its report. ``comm`` (3D): the whole input on
        every rank, the remesh on the ranks (volume.conform(comm=))."""
        d = X.shape[1]
        L = self.source(X, T, values)
        T32 = np.ascontiguousarray(T, np.int32)
        kw = self._kw()
        if comm is not None:
            if d != 3:
                raise ValueError("ShapeForestRemesher.conform(comm=): 3D only (2D on ranks conforms each rank's own "
                                 "cells: level_set)")
            kw.update(comm=sm.mpi.comm(comm), rounds=int(self.rounds))
        if d == 3:
            # the rule's floor and conform's tuning knobs go through options; planar's floor is fixed (2 h_min) and its
            # knobs are keywords
            options = dict(self.options)
            if self.no_cross_edges and self.cross_floor is not None:
                options.setdefault("cross_floor", float(self.cross_floor))
            if options:
                kw["options"] = options
            base = sm.Mesh.from_arrays(X, T32.astype(np.uint32), cell_type="tetra")
            return sm.volume.conform(base, None, L, **kw)
        kw.update(self.options)
        base = sm.Mesh.from_arrays(X, T32.astype(np.uint32))
        return sm.planar.conform(base, None, L, **kw)

    def level_set(self, msh, phi, facet_tags, cell_tags, params: Parameters, work_dir, name) -> RemeshedLevelSet:
        """phi's zero set made the mesh's interface: the new mesh with its tags and phi seeded by the tags (the loop
        redistances it)."""
        if msh.comm.size > 1:
            return self._level_set_ranks(msh, phi, params, name)
        X, T = cells_of(msh)
        values = np.ascontiguousarray(_vertex_values(phi), np.float64)
        t0 = time.perf_counter()
        try:
            out, report = self.conform(X, T, values)
            new, new_cells, new_facets = to_dolfinx(out, params, [ref for _, ref in self.patches], X.shape[1])
        except (sm.MeshingError, ValueError, RuntimeError) as error:   # the line search takes a RuntimeError as a rejected step
            self.calls.append(dict(name=name, seconds=time.perf_counter() - t0, cells=0, ok=False))
            if self.failures is not None:
                Path(self.failures).mkdir(parents=True, exist_ok=True)
                np.savez(Path(self.failures) / f"{name}.npz", points=X, cells=T, values=values, error=str(error))
            print(f"  remesh {name} failed: {type(error).__name__}: {error}"[:400], flush=True)
            raise RuntimeError(f"ShapeForest's conform failed at {name}") from error
        V = fem.functionspace(new, ("Lagrange", 1))
        new_phi = fem.Function(V, name="phi")
        _set_vertex_values(new_phi, _phase_seed_values(new_phi, new_cells, new_facets, params))
        seconds = time.perf_counter() - t0
        self.calls.append(dict(name=name, seconds=seconds, cells=int(new.topology.index_map(new.topology.dim).size_local),
                               ok=True))
        joined = (report.get("small_parts") or {}).get("joined") or []
        if joined:
            print(f"  remesh {name}: joined {len(joined)} parts below the resolution or min_part", flush=True)
        if report.get("fallback"):
            print(f"  remesh {name}: fell back to P1: {report['fallback']['reason']}"[:400], flush=True)
        return RemeshedLevelSet(new, new_phi, new_cells, new_facets)

    def _level_set_ranks(self, msh, phi, params: Parameters, name) -> RemeshedLevelSet:
        """On MPI ranks: each rank's own cells and phi's P1 values on them (``sm.sources.dolfinx_mesh``,
        ``from_fem(gather=False)``), conform on the ranks with nothing gathered (2D; 3D: ``volume.conform`` with
        ``parts``, ShapeForest's M2b), and the parts made one DOLFINx mesh (``sm.fem.to_dolfinx``) with sotutox's
        references. In 3D P1+G with ``parts=True`` goes the same way with G assembled on the ranks
        (``lumped_gradient_fem``): ShapeForest gathers the Hermite level set and cuts it on the parts, and its correction
        runs on the gathered volume, which dolfinx distributes from rank 0 (``to_dolfinx_from_whole``) until
        ShapeForest's M5. Otherwise P1+G, ``parts=False`` and changed_only take the whole input on every rank
        (``_level_set_ranks_3d``)."""
        d = msh.topology.dim
        parts = self.representation == "p1" if self.parts is None else bool(self.parts)
        if d == 3 and (self.changed_only or not parts):
            return self._level_set_ranks_3d(msh, phi, params, name)
        if self.representation != "p1" and d != 3:
            raise ValueError("ShapeForestRemesher on MPI ranks in 2D: representation='p1' (conform on ranks reads no "
                             "derivatives); valid: 'p1'")
        t0 = time.perf_counter()
        try:
            part = sm.sources.dolfinx_mesh(msh)
            if self.representation == "p1g":
                L = sm.sources.from_fem(phi, gather=False, dtype="float64", gradient=lumped_gradient_fem(phi),
                                        between="hermite")
            else:
                L = sm.sources.from_fem(phi, gather=False, dtype="float64")
            kw = self._kw()
            c = sm.mpi.comm(msh.comm)
            whole = d == 3 and self.representation == "p1g"   # the correction on the gathered volume (until M5)
            if d == 3:
                options = dict(self.options)
                if self.no_cross_edges and self.cross_floor is not None:
                    options.setdefault("cross_floor", float(self.cross_floor))
                if options:
                    kw["options"] = options
                mine, report = sm.volume.conform(part, None, L, comm=c, parts=True, gather=whole,
                                                 rounds=int(self.rounds), **kw)
            else:
                kw.update(self.options)
                mine, report = sm.planar.conform(part, None, L, comm=c, gather=False, **kw)
            if whole:
                new, new_cells, new_facets = to_dolfinx_from_whole(mine, params, [ref for _, ref in self.patches],
                                                                   msh.comm)
            else:
                out = sm.fem.to_dolfinx(mine, msh.comm)
                new, new_cells, new_facets = _tags_on_ranks(out, params, [ref for _, ref in self.patches])
        except (sm.MeshingError, ValueError, RuntimeError) as error:
            self.calls.append(dict(name=name, seconds=time.perf_counter() - t0, cells=0, ok=False))
            if msh.comm.rank == 0:
                print(f"  remesh {name} failed: {type(error).__name__}: {error}"[:400], flush=True)
            raise RuntimeError(f"ShapeForest's conform failed at {name}") from error
        V = fem.functionspace(new, ("Lagrange", 1))
        new_phi = fem.Function(V, name="phi")
        _set_vertex_values(new_phi, _phase_seed_values(new_phi, new_cells, new_facets, params))
        n_cells = new.comm.allreduce(int(new.topology.index_map(d).size_local))
        self.calls.append(dict(name=name, seconds=time.perf_counter() - t0, cells=n_cells, ok=True))
        return RemeshedLevelSet(new, new_phi, new_cells, new_facets)

    def _level_set_ranks_3d(self, msh, phi, params: Parameters, name) -> RemeshedLevelSet:
        """On MPI ranks in 3D (ShapeForest's docs/conform_mpi_plan.md sections 29 and 30, M2a and M4): the whole mesh and
        phi gathered on every rank (``gather_whole``), the cut and the correction there, the remesh on the ranks
        (``volume.conform(comm=)``; changed_only: its rounds start around the change); the result one DOLFINx mesh that
        dolfinx distributes from rank 0 (``to_dolfinx_from_whole``)."""
        t0 = time.perf_counter()
        X, T, values = gather_whole(msh, phi)
        try:
            out, report = self.conform(X, T, values, comm=msh.comm)
            new, new_cells, new_facets = to_dolfinx_from_whole(out, params, [ref for _, ref in self.patches], msh.comm)
        except (sm.MeshingError, ValueError, RuntimeError) as error:
            self.calls.append(dict(name=name, seconds=time.perf_counter() - t0, cells=0, ok=False))
            if msh.comm.rank == 0:
                if self.failures is not None:
                    Path(self.failures).mkdir(parents=True, exist_ok=True)
                    np.savez(Path(self.failures) / f"{name}.npz", points=X, cells=T, values=values, error=str(error))
                print(f"  remesh {name} failed: {type(error).__name__}: {error}"[:400], flush=True)
            raise RuntimeError(f"ShapeForest's conform failed at {name}") from error
        V = fem.functionspace(new, ("Lagrange", 1))
        new_phi = fem.Function(V, name="phi")
        _set_vertex_values(new_phi, _phase_seed_values(new_phi, new_cells, new_facets, params))
        n_cells = int(new.topology.index_map(3).size_global)
        self.calls.append(dict(name=name, seconds=time.perf_counter() - t0, cells=n_cells, ok=True))
        if msh.comm.rank == 0:
            joined = (report.get("small_parts") or {}).get("joined") or []
            if joined:
                print(f"  remesh {name}: joined {len(joined)} parts below the resolution or min_part", flush=True)
            if report.get("fallback"):
                print(f"  remesh {name}: fell back to P1: {report['fallback']['reason']}"[:400], flush=True)
        return RemeshedLevelSet(new, new_phi, new_cells, new_facets)


def _tags_on_ranks(out, params: Parameters, patch_refs: list[int]):
    """sotutox's references on a ``sm.fem.to_dolfinx`` result: material 0 solid, else void; the interface; the outer
    facets of patch k ``patch_refs[k]``, the other outer facets 0."""
    new = out.mesh
    d = new.topology.dim
    cells = np.asarray(out.cells.indices, np.int32)
    cell_tags = mesh.meshtags(new, d, cells, np.where(np.asarray(out.cells.values) == 0, params.ref_solid,
                                                       params.ref_void).astype(np.int32))
    indices, values = [], []
    if out.interface is not None:
        indices.append(np.asarray(out.interface.indices, np.int32))
        values.append(np.full(len(out.interface.indices), params.ref_interface, np.int32))
    if out.outer is not None:
        patch = np.asarray(out.outer.values)
        ok = (patch >= 0) & (patch < len(patch_refs))
        indices.append(np.asarray(out.outer.indices, np.int32)[ok])
        values.append(np.asarray(patch_refs, np.int32)[patch[ok]])
    new.topology.create_connectivity(d - 1, d)
    exterior = np.asarray(mesh.exterior_facet_indices(new.topology), np.int32)
    tagged = np.concatenate(indices) if indices else np.zeros(0, np.int32)
    rest = np.setdiff1d(exterior, tagged)
    indices.append(rest)
    values.append(np.zeros(len(rest), np.int32))
    idx, val = np.concatenate(indices), np.concatenate(values)
    idx, first = np.unique(idx, return_index=True)
    return new, cell_tags, mesh.meshtags(new, d - 1, idx, val[first])


def gather_whole(msh, phi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole mesh and phi's vertex values on every rank: each rank's nodes by their index in the mesh's input
    (``geometry.input_global_indices``, the same on every rank that holds a node) with their positions and values, and
    its owned cells by those indices. Positively oriented cells."""
    d = msh.topology.dim
    comm = msh.comm
    ign = np.asarray(msh.geometry.input_global_indices, np.int64)
    x = np.ascontiguousarray(msh.geometry.x[:, :d], np.float64)
    v = np.asarray(_vertex_values(phi), np.float64)
    own = msh.topology.index_map(d).size_local
    cells = ign[np.asarray(msh.geometry.dofmaps[0], np.int64)[:own]]
    parts = comm.allgather((ign, x, v, cells))
    n = 1 + max(int(p[0].max()) if len(p[0]) else -1 for p in parts)
    X = np.zeros((n, d))
    values = np.zeros(n)
    for ids, xs, vs, _ in parts:
        X[ids] = xs
        values[ids] = vs
    T = np.concatenate([p[3] for p in parts], axis=0)
    used = np.unique(T)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    X, values, T = X[used], values[used], remap[T]
    p = X[T]
    J = np.stack([p[:, j + 1] - p[:, 0] for j in range(d)], axis=2)
    flip = np.linalg.det(J) < 0.0
    T[flip] = T[flip][:, [1, 0] + list(range(2, d + 1))]
    return np.ascontiguousarray(X), np.ascontiguousarray(T), np.ascontiguousarray(values)


def to_dolfinx_from_whole(out, params: Parameters, patch_refs: list[int], comm):
    """A whole 3D conform result, the same on every rank, as one DOLFINx mesh distributed over ``comm`` (ShapeForest's
    ``sm.fem.from_parts``: rank 0 gives every cell, dolfinx's partitioner places them, with the ghosts ``dS`` needs) and
    sotutox's references: material 0 solid, else void; the interface ``params.ref_interface``; the outer faces of patch
    k ``patch_refs[k]``, the other outer faces 0."""
    X = np.asarray(out.points, np.float64)
    T = np.asarray(out.tets, np.int64)
    K = np.asarray(out.keys, np.uint64)
    refs = np.where(np.asarray(out.materials) == 0, params.ref_solid, params.ref_void).astype(np.int32)
    O = np.asarray(out.outer_faces, np.int64).reshape(-1, 3)
    ov = np.zeros(len(O), np.int32)
    if out.outer_face_patch is not None:
        patch = np.asarray(out.outer_face_patch)
        for k, ref in enumerate(patch_refs):
            ov[patch == k] = ref
    Fi = np.asarray(out.faces, np.int64).reshape(-1, 3)
    F = np.concatenate([O, Fi])
    Fv = np.concatenate([ov, np.full(len(Fi), params.ref_interface, np.int32)])
    if comm.rank != 0:
        X, T, K, refs, F, Fv = X[:0], T[:0], K[:0], refs[:0], F[:0], Fv[:0]
    msh, cell_tags, facets = sm.fem.from_parts(comm, X, T, K, cell_values=refs, facet_sets=dict(tags=(F, Fv)))
    return msh, cell_tags, facets["tags"]
