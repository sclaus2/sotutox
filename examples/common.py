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
    parser.add_argument("--mmg-binary", type=str, default=defaults.mmg_binary)
    parser.add_argument("--mmg-hmin", type=float, default=defaults.mmg_hmin)
    parser.add_argument("--mmg-hmax", type=float, default=defaults.mmg_hmax)
    parser.add_argument("--mmg-hausd", type=float, default=defaults.mmg_hausd)
    parser.add_argument("--mmg-hgrad", type=float, default=defaults.mmg_hgrad)
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


def run_example(
    params: Parameters,
    initial_level_set: Callable[[np.ndarray], np.ndarray],
    mark_boundaries: Callable[[mesh.Mesh, Parameters], BoundaryData],
    default_output_dir: Path,
    description: str,
    create_initial_mesh: Callable[[Parameters], mesh.Mesh] | None = None,
) -> list[dict]:
    """Run one named example using the shared optimization engine."""

    args = _parser(description, default_output_dir, params).parse_args()
    configured = replace(
        params,
        max_iterations=args.iterations,
        volume_target=(
            args.target_volume if args.target_volume is not None else params.volume_target
        ),
        output_every=max(1, args.output_every),
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
    )
    run_kwargs = {}
    if create_initial_mesh is not None:
        run_kwargs["create_initial_mesh"] = create_initial_mesh
    return run(
        configured,
        args.output_dir,
        initial_level_set=initial_level_set,
        mark_boundaries=mark_boundaries,
        write_xdmf=not args.no_xdmf,
        **run_kwargs,
    )
