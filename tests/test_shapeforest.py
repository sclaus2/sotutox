# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""ShapeForest as the loop's remesher (sotutox.shapeforest), skipped without ShapeForest:
- one 3D remesh call of the cantilever tags it as the mmg path does (solid and void, the interface, the clamp on x = 0,
  the load patch on x = 2) and keeps the box's volume;
- a short 2D loop runs with it, with no mmg anywhere (start "initial");
- the default remesher still calls the mmg functions the module holds when it runs (code that replaces them keeps
  working)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("shapeforest.mesh")
pytest.importorskip("shapeforest._sf_mesher")

from dolfinx import mesh  # noqa: E402

from examples import cantilever, cantilever_3d  # noqa: E402
from sotutox import optimization_mmg  # noqa: E402
from sotutox.geometry import initialize_level_set, scalar_space  # noqa: E402
from sotutox.shapeforest import ShapeForestRemesher, cells_of  # noqa: E402


def _measures(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    J = np.stack([X[T[:, j + 1]] - X[T[:, 0]] for j in range(d)], axis=2)
    return np.abs(np.linalg.det(J)) / (2.0 if d == 2 else 6.0)


def test_one_3d_remesh_tags_the_cantilever_as_the_mmg_path_does(tmp_path):
    params = cantilever_3d.CANTILEVER_3D_PARAMETERS
    msh = cantilever_3d.create_initial_mesh(params)
    boundaries = cantilever_3d.mark_boundaries(msh, params)
    settings = cantilever_3d.shapeforest_settings(params)
    settings.update(h=0.2, tol=0.05, h_min=0.1, start="initial")
    remesher = ShapeForestRemesher(**settings)
    first = remesher.standard(msh, boundaries.facet_tags, None, params, tmp_path, "initial_background")
    centre = np.array([1.0, 0.5, 0.5])
    phi = initialize_level_set(scalar_space(first.mesh), lambda x: 0.3 - np.linalg.norm(x - centre, axis=1))
    out = remesher.level_set(first.mesh, phi, first.facet_tags, first.cell_tags, params, tmp_path, "it0000_ls0")

    X, T = cells_of(out.mesh)
    refs = np.asarray(out.cell_tags.values)[np.argsort(np.asarray(out.cell_tags.indices))]
    assert set(np.unique(refs).tolist()) == {params.ref_solid, params.ref_void}
    vol = _measures(X, T)
    assert abs(vol.sum() - 2.0) < 1e-9
    void = vol[refs == params.ref_void].sum()
    assert abs(void - 4.0 / 3.0 * np.pi * 0.3**3) < 0.15 * 4.0 / 3.0 * np.pi * 0.3**3

    out.mesh.topology.create_connectivity(2, 0)
    facets, values = np.asarray(out.facet_tags.indices), np.asarray(out.facet_tags.values)
    P = mesh.entities_to_geometry(out.mesh, 2, facets.astype(np.int32))
    G = out.mesh.geometry.x
    clamp = P[values == params.ref_dirichlet]
    load = P[values == params.ref_neumann]
    assert len(clamp) > 0 and np.abs(G[clamp][:, :, 0]).max() < 1e-12
    assert len(load) > 0 and np.abs(G[load][:, :, 0] - 2.0).max() < 1e-12
    assert np.all((G[load][:, :, 1:] > 0.4 - 1e-9) & (G[load][:, :, 1:] < 0.6 + 1e-9))
    interface = P[values == params.ref_interface]
    assert len(interface) > 0
    r = np.linalg.norm(G[interface] - centre, axis=2)
    assert np.abs(r - 0.3).max() < 0.05
    assert remesher.calls[-1]["ok"]


def test_a_short_2d_loop_runs_with_shapeforest_and_no_mmg(tmp_path):
    params = replace(
        cantilever.CANTILEVER_PARAMETERS,
        max_iterations=2,
        initial_left_segments=20,
        initial_bottom_segments=40,
        initial_top_segments=40,
    )
    settings = cantilever.shapeforest_settings(params)
    settings.update(h=0.05, tol=0.0125, h_min=0.0125, start="initial", changed_only=False)
    remesher = ShapeForestRemesher(**settings)
    history = optimization_mmg.run(
        params,
        tmp_path,
        initial_level_set=cantilever.initial_level_set_values,
        mark_boundaries=cantilever.mark_boundaries,
        write_xdmf=False,
        remesher=remesher,
    )
    assert len(history) >= 2
    assert all(np.isfinite(float(row["compliance"])) for row in history if "compliance" in row)
    assert len(remesher.calls) >= 2 and remesher.calls[0]["ok"]


def test_the_default_remesher_calls_the_modules_mmg_functions_when_it_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(optimization_mmg, "run_mmg_level_set", lambda *args: seen.append("level_set") or "ls")
    monkeypatch.setattr(optimization_mmg, "run_mmg_standard", lambda *args: seen.append("standard") or "std")
    remesher = optimization_mmg.MmgRemesher()
    assert remesher.level_set(None, None, None, None, None, None, "x") == "ls"
    assert remesher.standard(None, None, None, None, None, "x") == "std"
    assert seen == ["level_set", "standard"]
