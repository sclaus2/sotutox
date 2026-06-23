# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotutox.redistance import graph_signed_distance


def test_graph_signed_distance_seeds_from_edge_crossing() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    values = np.array([-0.25, 0.25, 1.25])

    result = graph_signed_distance(points, edges, values)

    np.testing.assert_allclose(result, np.array([-0.5, 0.5, 1.5]))


def test_graph_signed_distance_preserves_exact_zero_vertex() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    values = np.array([-1.0, 0.0, 1.0])

    result = graph_signed_distance(points, edges, values)

    np.testing.assert_allclose(result, np.array([-1.0, 0.0, 1.0]))
