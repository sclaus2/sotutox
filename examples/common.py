# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared command-line handling for the example runners."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from dolfinx import mesh

from sotutox.mechanics import BoundaryData
from sotutox.optimization_mmg import run
from sotutox.parameters import Parameters


def _parser(
    description: str, default_output_dir: Path, defaults: Parameters
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--target-volume", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--output-every", type=int, default=1)
    parser.add_argument("--initial-left-segments", type=int, default=None)
    parser.add_argument("--initial-bottom-segments", type=int, default=None)
    parser.add_argument("--initial-top-segments", type=int, default=None)
    parser.add_argument("--mmg-binary", type=str, default=defaults.mmg_binary)
    parser.add_argument("--mmg-hmin", type=float, default=defaults.mmg_hmin)
    parser.add_argument("--mmg-hmax", type=float, default=defaults.mmg_hmax)
    parser.add_argument("--mmg-hausd", type=float, default=defaults.mmg_hausd)
    parser.add_argument("--mmg-hgrad", type=float, default=defaults.mmg_hgrad)
    parser.add_argument(
        "--remesher",
        choices=("mmg", "shapeforest"),
        default="mmg",
        help="The remesher: mmg (Medit files) or ShapeForest's conform (sotutox.shapeforest; the example's "
        "shapeforest_settings give its patches and sizes, the --sf-* options override them).",
    )
    parser.add_argument("--sf-h", type=float, default=None, help="ShapeForest: the mesh size h.")
    parser.add_argument("--sf-tol", type=float, default=None, help="ShapeForest: the interface's tolerance.")
    parser.add_argument("--sf-h-min", type=float, default=None, help="ShapeForest: h_min.")
    parser.add_argument(
        "--sf-level-set",
        choices=("p1", "p1g"),
        default=None,
        help="ShapeForest: phi's P1 values, or with G (the lumped projection of their gradient) a Hermite level set.",
    )
    parser.add_argument(
        "--sf-cross-floor",
        type=float,
        default=None,
        help="ShapeForest, 3D: no_cross_edges' floor x h_min (walls thinner than it hold no vertex).",
    )
    parser.add_argument(
        "--sf-start",
        choices=("mmg", "initial"),
        default=None,
        help="ShapeForest: the first mesh, mmg's remesh of the initial mesh or the initial mesh as it is.",
    )
    parser.add_argument(
        "--redistancing",
        choices=("auto", "graph", "exact"),
        default=None,
        help="How phi is rebuilt from the new mesh's tags: the distance along the edges (graph, one process), to the "
        "interface's facets (exact) or auto (graph on one process, exact on MPI ranks).",
    )
    parser.add_argument(
        "--sf-changed-only",
        choices=("on", "off"),
        default=None,
        help="ShapeForest: remesh only around the change (on) or everywhere (off; on MPI ranks in 3D it is off).",
    )
    parser.add_argument(
        "--sf-rounds",
        type=int,
        default=None,
        help="ShapeForest, 3D on MPI ranks: the remesh's rounds (each moves the parts' borders).",
    )
    parser.add_argument(
        "--advection",
        choices=("supg", "midpoint", "euler"),
        default=defaults.advection_backend,
        help="Native FEniCSx level-set advection backend.",
    )
    parser.add_argument(
        "--supg-tau-scale",
        type=float,
        default=defaults.supg_tau_scale,
        help="Multiplier for the SUPG stabilization parameter.",
    )
    parser.add_argument(
        "--topological-derivatives",
        dest="topological_derivatives",
        action="store_true",
        default=None,
        help="Enable sotuto/topder-style topological derivative hole creation.",
    )
    parser.add_argument(
        "--no-topological-derivatives",
        dest="topological_derivatives",
        action="store_false",
        help="Disable topological derivative hole creation.",
    )
    parser.add_argument(
        "--topological-interval",
        type=int,
        default=defaults.topological_interval,
        help="Apply topological derivatives every N iterations when enabled.",
    )
    parser.add_argument(
        "--topological-last-iteration",
        type=int,
        default=defaults.topological_last_iteration,
        help="Last iteration eligible for topological derivative hole creation.",
    )
    parser.add_argument(
        "--topological-volume-fraction",
        type=float,
        default=defaults.topological_volume_fraction,
        help="Fraction of current solid volume removed by one topological update.",
    )
    parser.add_argument(
        "--topological-min-distance-from-interface",
        type=float,
        default=defaults.topological_min_distance_from_interface,
        help="Only create topological holes where phi is below this negative clearance.",
    )
    parser.add_argument(
        "--topological-min-hole-radius",
        type=float,
        default=defaults.topological_min_hole_radius,
        help="Reject newly-created topological hole components below this radius.",
    )
    parser.add_argument(
        "--void-cleanup-min-hole-radius",
        type=float,
        default=defaults.void_cleanup_min_hole_radius,
        help=(
            "Remove accepted void components below this equivalent radius; "
            "0 disables cleanup."
        ),
    )
    parser.add_argument("--no-xdmf", action="store_true")
    return parser


def _shapeforest_remesher(args, params: Parameters, settings: Callable[[Parameters], dict] | None):
    """The example's ShapeForest remesher, its settings overridden by the --sf-* options."""
    if settings is None:
        raise ValueError(
            "--remesher shapeforest: this example defines no shapeforest_settings (its patches); valid: --remesher mmg"
        )
    try:
        from sotutox.shapeforest import ShapeForestRemesher
    except ImportError as error:
        raise ImportError(
            f"--remesher shapeforest needs ShapeForest (with its compiled mesher) on the path: {error}"
        ) from error
    kw = dict(settings(params))
    for option, key in (
        ("sf_h", "h"),
        ("sf_tol", "tol"),
        ("sf_h_min", "h_min"),
        ("sf_level_set", "representation"),
        ("sf_cross_floor", "cross_floor"),
        ("sf_start", "start"),
        ("sf_rounds", "rounds"),
    ):
        if getattr(args, option) is not None:
            kw[key] = getattr(args, option)
    if args.sf_changed_only is not None:
        kw["changed_only"] = args.sf_changed_only == "on"
    if args.sf_h is not None and args.sf_tol is None:
        kw["tol"] = 0.25 * args.sf_h
    kw.setdefault("failures", args.output_dir / "failures")
    return ShapeForestRemesher(**kw)


def run_example(
    params: Parameters,
    initial_level_set: Callable[[np.ndarray], np.ndarray],
    mark_boundaries: Callable[[mesh.Mesh, Parameters], BoundaryData],
    default_output_dir: Path,
    description: str,
    create_initial_mesh: Callable[[Parameters], mesh.Mesh] | None = None,
    run_kwargs: dict | None = None,
    driver: Callable = run,
    shapeforest_settings: Callable[[Parameters], dict] | None = None,
) -> list[dict]:
    """Run one named example using the shared optimization engine. ``shapeforest_settings``: the example's
    ``ShapeForestRemesher`` settings (patches, sizes) for ``--remesher shapeforest``."""

    args = _parser(description, default_output_dir, params).parse_args()
    configured = replace(
        params,
        max_iterations=args.iterations,
        volume_target=(
            args.target_volume if args.target_volume is not None else params.volume_target
        ),
        output_every=max(1, args.output_every),
        initial_left_segments=(
            args.initial_left_segments
            if args.initial_left_segments is not None
            else params.initial_left_segments
        ),
        initial_bottom_segments=(
            args.initial_bottom_segments
            if args.initial_bottom_segments is not None
            else params.initial_bottom_segments
        ),
        initial_top_segments=(
            args.initial_top_segments
            if args.initial_top_segments is not None
            else params.initial_top_segments
        ),
        mmg_binary=args.mmg_binary,
        mmg_hmin=args.mmg_hmin,
        mmg_hmax=args.mmg_hmax,
        mmg_hausd=args.mmg_hausd,
        mmg_hgrad=args.mmg_hgrad,
        advection_backend=args.advection,
        supg_tau_scale=args.supg_tau_scale,
        topological_derivatives=(
            params.topological_derivatives
            if args.topological_derivatives is None
            else args.topological_derivatives
        ),
        topological_interval=args.topological_interval,
        topological_last_iteration=args.topological_last_iteration,
        topological_volume_fraction=args.topological_volume_fraction,
        topological_min_distance_from_interface=(
            args.topological_min_distance_from_interface
        ),
        topological_min_hole_radius=args.topological_min_hole_radius,
        void_cleanup_min_hole_radius=args.void_cleanup_min_hole_radius,
        redistancing=args.redistancing if args.redistancing is not None else params.redistancing,
    )
    configured_run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
    if create_initial_mesh is not None:
        configured_run_kwargs["create_initial_mesh"] = create_initial_mesh
    if args.remesher == "shapeforest":
        configured_run_kwargs["remesher"] = _shapeforest_remesher(args, configured, shapeforest_settings)
    return driver(
        configured,
        args.output_dir,
        initial_level_set=initial_level_set,
        mark_boundaries=mark_boundaries,
        write_xdmf=not args.no_xdmf,
        **configured_run_kwargs,
    )
