# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The loop on MPI ranks with ShapeForest's remesher, through mpiexec on 1 and 2 ranks:
- 2D: the cantilever, three iterations (each rank conforms its own cells);
- 3D: the cantilever at h 0.12, two iterations (every rank is given the whole input, the remesh runs on the ranks),
  changed_only off on both, as on ranks.
Both runs finish, write one history (rank 0), and follow each other: a mesh on ranks is a different mesh, so J agrees
to a few parts in a thousand, not bit for bit.

Needs ShapeForest with an MPI build of its mesher (SHAPEFOREST_MESHER_PATH names it, as ShapeForest's tests/mpi/run.py
does) and the mpiexec next to this Python; skipped otherwise."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

sm = pytest.importorskip("shapeforest.mesh")   # its loader picks the mesher (SHAPEFOREST_MESHER_PATH): no direct import

ROOT = Path(__file__).resolve().parents[1]
MPIEXEC = Path(sys.executable).with_name("mpiexec")


def _have_mpi_mesher() -> bool:
    try:
        return bool(sm.mpi.have_mpi())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not MPIEXEC.exists() or not _have_mpi_mesher(),
    reason="needs mpiexec next to Python and an MPI build of ShapeForest's mesher (SHAPEFOREST_MESHER_PATH)",
)


def _run(ranks: int, out: Path, example: str = "cantilever.py", options: tuple = ("--sf-level-set", "p1"),
         iterations: int = 3) -> list[dict]:
    env = dict(os.environ, FI_PROVIDER="tcp", OMP_NUM_THREADS="1", SFM_THREADS="1", MPIEXEC_TIMEOUT="600",
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [str(MPIEXEC), "-n", str(ranks), sys.executable, str(ROOT / "examples" / example), "--remesher",
           "shapeforest", "--sf-start", "initial", *options, "--iterations", str(iterations), "--output-dir", str(out),
           "--no-xdmf"]
    proc = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with (out / "history.csv").open() as f:
        return [row for row in csv.DictReader(f) if row["accepted"] in ("1", "True", "true")]


def test_the_2d_loop_runs_on_two_ranks_and_follows_one(tmp_path):
    one = _run(1, tmp_path / "one")
    two = _run(2, tmp_path / "two")
    assert len(one) >= 3 and len(two) >= 3
    for a, b in zip(one, two):
        assert abs(float(a["compliance"]) - float(b["compliance"])) < 5e-3 * float(a["compliance"])
    assert float(two[-1]["compliance"]) < float(two[0]["compliance"])


def test_the_3d_loop_runs_on_two_ranks_and_follows_one(tmp_path):
    options = ("--sf-h", "0.12", "--sf-h-min", "0.06", "--sf-changed-only", "off")
    one = _run(1, tmp_path / "one", "cantilever_3d.py", options, iterations=2)
    two = _run(2, tmp_path / "two", "cantilever_3d.py", options, iterations=2)
    assert len(one) >= 2 and len(two) >= 2
    for a, b in zip(one, two):
        assert abs(float(a["compliance"]) - float(b["compliance"])) < 1e-2 * float(a["compliance"])
        assert abs(float(a["volume"]) - float(b["volume"])) < 2e-3 * float(a["volume"])
    assert float(two[-1]["volume"]) < float(two[0]["volume"])
