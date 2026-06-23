# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MMG/Medit remeshing helpers for fitted level-set optimization.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import basix.ufl
import meshio
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.spatial import cKDTree

from .parameters import Parameters
from .redistance import graph_signed_distance


@dataclass(slots=True)
class RemeshedLevelSet:
    """DOLFINx view of an MMG-remeshed level-set mesh."""

    mesh: mesh.Mesh
    phi: fem.Function
    cell_tags: mesh.MeshTags
    facet_tags: mesh.MeshTags


@dataclass(slots=True)
class RemeshedMesh:
    """DOLFINx view of a standard MMG-remeshed mesh."""

    mesh: mesh.Mesh
    cell_tags: mesh.MeshTags
    facet_tags: mesh.MeshTags


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


def _mesh_edges(msh: mesh.Mesh) -> np.ndarray:
    """Return local mesh edges as pairs of geometry vertex indices."""

    edim = 1
    msh.topology.create_connectivity(edim, 0)
    e_to_v = msh.topology.connectivity(edim, 0)
    num_edges = msh.topology.index_map(edim).size_local
    return np.asarray(
        [e_to_v.links(edge) for edge in range(num_edges)], dtype=np.int64
    )


def _tag_dict(tags: mesh.MeshTags | None) -> dict[int, int]:
    if tags is None:
        return {}
    return {int(entity): int(value) for entity, value in zip(tags.indices, tags.values)}


def _cell_refs(msh: mesh.Mesh, cell_tags: mesh.MeshTags | None) -> np.ndarray:
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    refs = np.ones(num_cells, dtype=np.int32)
    if cell_tags is not None:
        refs[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
            cell_tags.values, dtype=np.int32
        )
    return refs


def _phase_vertex_masks(
    msh: mesh.Mesh, refs: np.ndarray, params: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    cells = _cell_vertices(msh)
    num_vertices = msh.geometry.x.shape[0]
    touches_solid = np.zeros(num_vertices, dtype=bool)
    touches_void = np.zeros(num_vertices, dtype=bool)

    for cell_vertices in cells[refs == params.ref_solid]:
        touches_solid[cell_vertices] = True
    for cell_vertices in cells[refs == params.ref_void]:
        touches_void[cell_vertices] = True
    return touches_solid, touches_void


def _input_facets(
    msh: mesh.Mesh, facet_tags: mesh.MeshTags | None
) -> tuple[np.ndarray, np.ndarray]:
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    exterior = mesh.exterior_facet_indices(msh.topology)
    tagged = np.asarray([], dtype=np.int32) if facet_tags is None else facet_tags.indices
    facets = np.unique(np.concatenate([exterior, tagged]).astype(np.int32))
    values = np.zeros(len(facets), dtype=np.int32)
    tags = _tag_dict(facet_tags)
    for i, facet in enumerate(facets):
        values[i] = tags.get(int(facet), 0)
    return facets, values


def _vertex_values(function: fem.Function) -> np.ndarray:
    """Return scalar P1 values in mesh-vertex order."""

    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, dofs = cKDTree(coords).query(vertex_coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match P1 dofs to mesh vertices.")
    return np.asarray(function.x.array[dofs], dtype=np.float64)


def _set_vertex_values(function: fem.Function, values: np.ndarray) -> None:
    """Assign scalar P1 values provided in mesh-vertex order."""

    V = function.function_space
    coords = V.tabulate_dof_coordinates()[:, : V.mesh.geometry.dim]
    vertex_coords = V.mesh.geometry.x[:, : V.mesh.geometry.dim]
    distances, vertices = cKDTree(vertex_coords).query(coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match mesh vertices to P1 dofs.")
    function.x.array[:] = np.asarray(values, dtype=np.float64)[vertices]
    function.x.scatter_forward()


def _interface_facets(
    facet_tags: mesh.MeshTags, params: Parameters
) -> np.ndarray:
    values = np.asarray(facet_tags.values, dtype=np.int32)
    return np.asarray(facet_tags.indices, dtype=np.int32)[
        values == params.ref_interface
    ]


def _phase_seed_values(
    phi: fem.Function,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
) -> np.ndarray:
    """Return sign/zero seeds from MMG phase and interface tags."""

    msh = phi.function_space.mesh
    refs = _cell_refs(msh, cell_tags)
    touches_solid, touches_void = _phase_vertex_masks(msh, refs, params)
    values = _vertex_values(phi)
    seeds = np.where(values < 0.0, -1.0, 1.0)

    solid_only = touches_solid & ~touches_void
    void_only = touches_void & ~touches_solid
    interface = touches_solid & touches_void
    seeds[solid_only] = -1.0
    seeds[void_only] = 1.0
    seeds[interface] = 0.0

    facets = _interface_facets(facet_tags, params)
    if len(facets) > 0:
        seeds[np.unique(_facet_vertices(msh, facets).reshape(-1))] = 0.0
    return seeds


def write_medit_mesh_2d(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_refs: np.ndarray,
    edges: np.ndarray,
    edge_refs: np.ndarray,
) -> None:
    """Write a linear 2D Medit mesh file accepted by MMG."""

    with path.open("w") as f:
        f.write("MeshVersionFormatted 2\nDimension 2\n\n")
        f.write(f"Vertices\n{len(points)}\n")
        for p in points:
            f.write(f"{p[0]:.17g} {p[1]:.17g} 0\n")
        f.write(f"\nEdges\n{len(edges)}\n")
        for edge, ref in zip(edges, edge_refs):
            f.write(f"{edge[0] + 1} {edge[1] + 1} {int(ref)}\n")
        f.write(f"\nTriangles\n{len(triangles)}\n")
        for tri, ref in zip(triangles, triangle_refs):
            f.write(f"{tri[0] + 1} {tri[1] + 1} {tri[2] + 1} {int(ref)}\n")
        f.write("\nEnd\n")


def write_medit_mesh_3d(
    path: Path,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    tetra_refs: np.ndarray,
    triangles: np.ndarray,
    triangle_refs: np.ndarray,
) -> None:
    """Write a linear 3D Medit mesh file accepted by MMG."""

    with path.open("w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n\n")
        f.write(f"Vertices\n{len(points)}\n")
        for p in points:
            f.write(f"{p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 0\n")
        f.write(f"\nTriangles\n{len(triangles)}\n")
        for tri, ref in zip(triangles, triangle_refs):
            f.write(f"{tri[0] + 1} {tri[1] + 1} {tri[2] + 1} {int(ref)}\n")
        f.write(f"\nTetrahedra\n{len(tetrahedra)}\n")
        for tet, ref in zip(tetrahedra, tetra_refs):
            f.write(
                f"{tet[0] + 1} {tet[1] + 1} {tet[2] + 1} {tet[3] + 1} "
                f"{int(ref)}\n"
            )
        f.write("\nEnd\n")


def write_medit_sol(path: Path, values: np.ndarray, dimension: int) -> None:
    """Write a scalar P1 Medit solution file accepted by MMG."""

    with path.open("w") as f:
        f.write(f"MeshVersionFormatted 2\nDimension {dimension}\n\n")
        f.write(f"SolAtVertices\n{len(values)}\n1 1\n")
        for value in values:
            f.write(f"{float(value):.17g}\n")
        f.write("\nEnd\n")


def read_medit_sol(path: Path) -> np.ndarray:
    """Read scalar values from a Medit ``.sol`` file written by MMG."""

    tokens = path.read_text().split()
    try:
        idx = tokens.index("SolAtVertices")
    except ValueError as exc:
        raise RuntimeError(f"{path} does not contain SolAtVertices") from exc
    count = int(tokens[idx + 1])
    num_components = int(tokens[idx + 2])
    if num_components != 1:
        raise RuntimeError(f"{path} contains {num_components} solution components, expected 1")
    # tokens[idx + 3] is the Medit component type (1 == scalar).
    start = idx + 4
    return np.asarray([float(v) for v in tokens[start : start + count]], dtype=np.float64)


def write_dolfinx_medit_input(
    msh: mesh.Mesh,
    phi: fem.Function,
    facet_tags: mesh.MeshTags | None,
    cell_tags: mesh.MeshTags | None,
    mesh_path: Path,
    sol_path: Path,
) -> None:
    """Write the current DOLFINx mesh and level set for ``mmg2d -ls``."""

    write_dolfinx_medit_mesh(msh, facet_tags, cell_tags, mesh_path)
    write_medit_sol(sol_path, _vertex_values(phi), msh.topology.dim)


def write_dolfinx_medit_mesh(
    msh: mesh.Mesh,
    facet_tags: mesh.MeshTags | None,
    cell_tags: mesh.MeshTags | None,
    mesh_path: Path,
) -> None:
    """Write the current DOLFINx mesh for standard ``mmg2d`` remeshing."""

    facets, facet_refs = _input_facets(msh, facet_tags)
    if msh.topology.dim == 2:
        points = msh.geometry.x[:, :2]
        triangles = _cell_vertices(msh)
        edges = _facet_vertices(msh, facets)
        write_medit_mesh_2d(
            mesh_path,
            points,
            triangles,
            _cell_refs(msh, cell_tags),
            edges,
            facet_refs,
        )
        return

    if msh.topology.dim == 3:
        points = msh.geometry.x[:, :3]
        tetrahedra = _cell_vertices(msh)
        triangles = _facet_vertices(msh, facets)
        write_medit_mesh_3d(
            mesh_path,
            points,
            tetrahedra,
            _cell_refs(msh, cell_tags),
            triangles,
            facet_refs,
        )
        return

    raise RuntimeError("MMG remeshing is only implemented for 2D and 3D meshes.")


def _meshio_cells(medit: meshio.Mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    for i, block in enumerate(medit.cells):
        if block.type == cell_type:
            refs = np.asarray(medit.cell_data["medit:ref"][i], dtype=np.int32)
            return np.asarray(block.data, dtype=np.int64), refs
    widths = {"line": 2, "triangle": 3, "tetra": 4}
    return (
        np.empty((0, widths[cell_type]), dtype=np.int64),
        np.empty(0, dtype=np.int32),
    )


def _entity_key(points: np.ndarray, entity: np.ndarray) -> tuple[tuple[float, ...], ...]:
    vertices = [tuple(np.round(points[int(v)], 12)) for v in entity]
    return tuple(sorted(vertices))


def _cell_tags_from_cells(
    msh: mesh.Mesh,
    cells_in: np.ndarray,
    cell_refs: np.ndarray,
    input_points: np.ndarray,
) -> mesh.MeshTags:
    points = msh.geometry.x[:, : msh.geometry.dim]
    by_vertices = {
        _entity_key(input_points, cell): int(ref)
        for cell, ref in zip(cells_in, cell_refs)
    }
    cells = _cell_vertices(msh)
    entities: list[int] = []
    values: list[int] = []
    for cell, vertices in enumerate(cells):
        key = _entity_key(points, vertices)
        if key not in by_vertices:
            raise RuntimeError("Could not map an MMG cell reference to DOLFINx.")
        entities.append(cell)
        values.append(by_vertices[key])
    return mesh.meshtags(
        msh,
        msh.topology.dim,
        np.asarray(entities, dtype=np.int32),
        np.asarray(values, dtype=np.int32),
    )


def _facet_tags_from_boundary_cells(
    msh: mesh.Mesh,
    boundary_cells: np.ndarray,
    boundary_refs: np.ndarray,
    input_points: np.ndarray,
) -> mesh.MeshTags:
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    f_to_v = msh.topology.connectivity(fdim, 0)
    num_facets = msh.topology.index_map(fdim).size_local
    points = msh.geometry.x[:, : msh.geometry.dim]
    by_vertices = {
        _entity_key(points, f_to_v.links(facet)): facet
        for facet in range(num_facets)
    }
    entities: list[int] = []
    values: list[int] = []
    for boundary, ref in zip(boundary_cells, boundary_refs):
        key = _entity_key(input_points, boundary)
        if key in by_vertices:
            entities.append(by_vertices[key])
            values.append(int(ref))
    order = np.argsort(np.asarray(entities, dtype=np.int32))
    return mesh.meshtags(
        msh,
        fdim,
        np.asarray(entities, dtype=np.int32)[order],
        np.asarray(values, dtype=np.int32)[order],
    )


def read_mmg_mesh_output(mesh_path: Path, params: Parameters) -> RemeshedMesh:
    """Read a standard MMG Medit output mesh into DOLFINx mesh and tags."""

    medit = meshio.read(mesh_path)
    tetrahedra, tetra_refs = _meshio_cells(medit, "tetra")
    if len(tetrahedra) > 0:
        triangles, triangle_refs = _meshio_cells(medit, "triangle")
        points = np.asarray(medit.points[:, :3], dtype=np.float64)
        domain = ufl.Mesh(
            basix.ufl.element(
                "Lagrange", "tetrahedron", 1, shape=(3,), dtype=np.float64
            )
        )
        msh = mesh.create_mesh(MPI.COMM_WORLD, tetrahedra, domain, points)
        cell_tags = _cell_tags_from_cells(msh, tetrahedra, tetra_refs, points)
        facet_tags = _facet_tags_from_boundary_cells(
            msh, triangles, triangle_refs, points
        )
        return RemeshedMesh(msh, cell_tags, facet_tags)

    triangles, triangle_refs = _meshio_cells(medit, "triangle")
    lines, line_refs = _meshio_cells(medit, "line")
    if len(triangles) == 0:
        raise RuntimeError(f"MMG output {mesh_path} contains no cells")

    points = np.asarray(medit.points[:, :2], dtype=np.float64)
    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=np.float64)
    )
    msh = mesh.create_mesh(MPI.COMM_WORLD, triangles, domain, points)
    cell_tags = _cell_tags_from_cells(msh, triangles, triangle_refs, points)
    facet_tags = _facet_tags_from_boundary_cells(msh, lines, line_refs, points)
    return RemeshedMesh(msh, cell_tags, facet_tags)


def read_mmg_output(mesh_path: Path, sol_path: Path, params: Parameters) -> RemeshedLevelSet:
    """Read MMG Medit output into DOLFINx mesh, tags, and P1 level set."""

    remeshed = read_mmg_mesh_output(mesh_path, params)
    msh = remeshed.mesh

    V = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V, name="phi")
    values = read_medit_sol(sol_path)
    gdim = msh.geometry.dim
    medit_points = np.asarray(meshio.read(mesh_path).points[:, :gdim], dtype=np.float64)
    if len(values) != len(medit_points):
        raise RuntimeError(
            f"MMG solution {sol_path} has {len(values)} values for {len(medit_points)} vertices"
        )
    coords = V.tabulate_dof_coordinates()[:, :gdim]
    distances, point_ids = cKDTree(medit_points).query(coords)
    if np.max(distances) > 1.0e-10:
        raise RuntimeError("Could not match MMG solution vertices to DOLFINx P1 dofs.")
    phi.x.array[:] = values[point_ids]
    phi.x.scatter_forward()
    result = RemeshedLevelSet(msh, phi, remeshed.cell_tags, remeshed.facet_tags)
    orient_level_set_from_cell_tags(result.phi, result.cell_tags, params)
    return result


def orient_level_set_from_cell_tags(
    phi: fem.Function,
    cell_tags: mesh.MeshTags,
    params: Parameters,
    *,
    zero_interface: bool = False,
) -> None:
    """Enforce the sotuto convention ``phi < 0`` in solid cells.

    MMG's cell labels reliably identify the two phases after ``-ls``. The
    accompanying ``.sol`` may be unsigned, so this function reconstructs vertex
    signs from adjacent solid/void cell references.
    """

    msh = phi.function_space.mesh
    tdim = msh.topology.dim
    refs = np.ones(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    refs[np.asarray(cell_tags.indices, dtype=np.int32)] = np.asarray(
        cell_tags.values, dtype=np.int32
    )
    values = np.abs(_vertex_values(phi))
    solid = refs == params.ref_solid
    void = refs == params.ref_void
    if not np.any(solid) or not np.any(void):
        return

    touches_solid, touches_void = _phase_vertex_masks(msh, refs, params)

    signed = values.copy()
    solid_only = touches_solid & ~touches_void
    void_only = touches_void & ~touches_solid
    interface = touches_solid & touches_void
    signed[solid_only] = -values[solid_only]
    signed[void_only] = values[void_only]
    signed[interface] = 0.0 if zero_interface else values[interface]
    _set_vertex_values(phi, signed)


def distance_level_set_from_tags(
    phi: fem.Function,
    cell_tags: mesh.MeshTags,
    facet_tags: mesh.MeshTags,
    params: Parameters,
) -> None:
    """Rebuild a signed interface-distance field from MMG labels.

    This is the fitted-mesh analogue of sotuto's ``mshdist -dom`` step. It
    builds sign/zero seeds from adjacent MMG cell references and interface
    facets, then propagates them with the native graph signed-distance helper.
    """

    msh = phi.function_space.mesh
    if msh.comm.size != 1:
        raise RuntimeError("Graph redistancing is serial-only.")
    signed = graph_signed_distance(
        msh.geometry.x[:, : msh.geometry.dim],
        _mesh_edges(msh),
        _phase_seed_values(phi, cell_tags, facet_tags, params),
    )
    _set_vertex_values(phi, signed)


def run_mmg_standard(
    msh: mesh.Mesh,
    facet_tags: mesh.MeshTags | None,
    cell_tags: mesh.MeshTags | None,
    params: Parameters,
    work_dir: Path,
    name: str,
) -> RemeshedMesh:
    """Run the first standard MMG remesh used by ``sotuto``."""

    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir.resolve()
    in_mesh = work_dir / f"{name}.mesh"
    out_mesh = work_dir / f"{name}.out.mesh"
    write_dolfinx_medit_mesh(msh, facet_tags, cell_tags, in_mesh)

    cmd = [
        params.mmg_binary,
        "-in",
        str(in_mesh),
        "-hmin",
        str(params.mmg_hmin),
        "-hmax",
        str(params.mmg_hmax),
        "-hausd",
        str(params.mmg_hausd),
        "-hgrad",
        str(params.mmg_hgrad),
        "-rmc",
        "-out",
        str(out_mesh),
    ]
    proc = subprocess.run(cmd, cwd=work_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "MMG standard remeshing failed\n"
            + "Command: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )
    if not out_mesh.exists():
        raise RuntimeError(f"MMG did not write expected mesh file {out_mesh}")
    return read_mmg_mesh_output(out_mesh, params)


def run_mmg_level_set(
    msh: mesh.Mesh,
    phi: fem.Function,
    facet_tags: mesh.MeshTags | None,
    cell_tags: mesh.MeshTags | None,
    params: Parameters,
    work_dir: Path,
    name: str,
) -> RemeshedLevelSet:
    """Run MMG in level-set mode and return DOLFINx data."""

    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir.resolve()
    in_mesh = work_dir / f"{name}.mesh"
    in_sol = work_dir / f"{name}.sol"
    out_mesh = work_dir / f"{name}.out.mesh"
    out_sol = work_dir / f"{name}.out.sol"
    default_name = "DEFAULT.mmg3d" if msh.topology.dim == 3 else "DEFAULT.mmg2d"
    default_path = work_dir / default_name
    default_path.write_text(
        f"LSBaseReferences\n1\n\n{int(params.ref_dirichlet)}\n"
    )
    write_dolfinx_medit_input(msh, phi, facet_tags, cell_tags, in_mesh, in_sol)

    cmd = [
        params.mmg_binary,
        "-in",
        str(in_mesh),
        "-sol",
        str(in_sol),
        "-ls",
        "0",
        "-hmin",
        str(params.mmg_hmin),
        "-hmax",
        str(params.mmg_hmax),
        "-hausd",
        str(params.mmg_hausd),
        "-hgrad",
        str(params.mmg_hgrad),
        "-nr",
        "-rmc",
        "-out",
        str(out_mesh),
    ]
    proc = subprocess.run(cmd, cwd=work_dir, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "MMG remeshing failed\n"
            + "Command: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )
    if not out_sol.exists():
        raise RuntimeError(f"MMG did not write expected solution file {out_sol}")
    return read_mmg_output(out_mesh, out_sol, params)
