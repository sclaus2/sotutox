# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import l_beam, multiload_bridge


def test_l_beam_level_set_matches_stress_hole_sign() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.17, 0.25],
            [1.5, 1.5],
        ],
        dtype=float,
    )

    phi = l_beam.initial_level_set_values(coords)

    assert phi[0] > 0.0
    assert phi[1] > 0.0
    assert phi[2] < 0.0


def test_multiload_bridge_level_set_matches_constraints_holes() -> None:
    coords = np.array(
        [
            [0.0, 1.5],
            [0.5, 1.0],
            [5.0, 0.05],
            [5.0, 1.8],
        ],
        dtype=float,
    )

    phi = multiload_bridge.initial_level_set_values(coords)

    assert phi[0] > 0.0
    assert phi[1] > 0.0
    assert phi[2] > 0.0
    assert phi[3] < 0.0


def test_multiload_bridge_load_intervals_match_constraints_count() -> None:
    intervals = multiload_bridge._load_intervals(
        multiload_bridge.MULTILOAD_BRIDGE_PARAMETERS
    )

    assert len(intervals) == multiload_bridge.MULTILOAD_NUM_LOAD_CASES
    assert intervals[0][1] == multiload_bridge.MULTILOAD_BRIDGE_PARAMETERS.length
