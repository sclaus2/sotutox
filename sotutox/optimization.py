# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Null-space descent helpers for the body-fitted sotuto port.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .mechanics import Gradients, State, regularized_inner
from .parameters import Parameters


@dataclass(slots=True)
class DescentData:
    """Scalar fields and coefficients from the null-space descent formula."""

    speed: fem.Function
    lagrange: float
    penalty: float
    alpha_objective: float
    alpha_constraint: float
    norm_xi_objective: float


def _max_abs(function: fem.Function) -> float:
    local = float(np.max(np.abs(function.x.array))) if len(function.x.array) else 0.0
    return float(function.function_space.mesh.comm.allreduce(local, op=MPI.MAX))


def _linear_combination(
    V: fem.FunctionSpace, terms: list[tuple[float, fem.Function]], name: str
) -> fem.Function:
    result = fem.Function(V, name=name)
    result.x.array[:] = 0.0
    for coefficient, function in terms:
        result.x.array[:] += coefficient * function.x.array
    result.x.scatter_forward()
    return result


def compute_descent(
    gradients: Gradients,
    state: State,
    params: Parameters,
    max_norm_xi_objective: float,
) -> DescentData:
    """Reproduce the null-space constrained descent from ``descent.edp``."""

    return compute_descent_with_inner(
        gradients,
        state,
        params,
        max_norm_xi_objective,
        lambda f, g: regularized_inner(f, g, params),
    )


def compute_descent_with_inner(
    gradients: Gradients,
    state: State,
    params: Parameters,
    max_norm_xi_objective: float,
    inner: Callable[[fem.Function, fem.Function], float],
) -> DescentData:
    """Compute the null-space descent using an example-specific inner product."""

    V = gradients.compliance.function_space
    m = inner(gradients.volume, gradients.volume)
    if abs(m) < params.eps:
        raise RuntimeError("Volume gradient has near-zero regularized norm.")

    lagrange = inner(gradients.compliance, gradients.volume) / m
    xi_objective = _linear_combination(
        V,
        [(1.0, gradients.compliance), (-lagrange, gradients.volume)],
        "xi_objective",
    )
    xi_constraint = _linear_combination(
        V,
        [((state.volume - params.target_volume) / m, gradients.volume)],
        "xi_constraint",
    )

    norm_xi_objective = _max_abs(xi_objective)
    norm_xi_constraint = _max_abs(xi_constraint)
    normalized_objective = max(norm_xi_objective, max_norm_xi_objective)
    alpha_objective = (
        params.objective_weight
        * params.mesh_size
        / (params.eps**2 + normalized_objective)
    )
    alpha_constraint = (
        params.constraint_weight
        * params.mesh_size
        / (params.eps**2 + norm_xi_constraint)
    )
    if params.constraint_cap is not None:
        alpha_constraint = min(alpha_constraint, params.constraint_cap)

    speed = _linear_combination(
        V,
        [
            (-alpha_objective, xi_objective),
            (-alpha_constraint, xi_constraint),
        ],
        "normal_speed",
    )
    return DescentData(
        speed=speed,
        lagrange=lagrange,
        penalty=m,
        alpha_objective=alpha_objective,
        alpha_constraint=alpha_constraint,
        norm_xi_objective=norm_xi_objective,
    )


def merit(state: State, descent: DescentData, params: Parameters) -> float:
    """Merit function used by the original line search."""

    volume_error = state.volume - params.target_volume
    return (
        descent.alpha_objective
        * (state.compliance - descent.lagrange * volume_error)
        + 0.5 * descent.alpha_constraint / descent.penalty * volume_error**2
    )
