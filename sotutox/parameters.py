# SPDX-FileCopyrightText: 2026 ONERA
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generic parameter container for the MMG-remeshed optimization driver.

Algorithmic reference: https://github.com/dapogny/sotuto by Charles Dapogny
and collaborators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Parameters:
    """Numerical constants used by the generic optimization loop."""

    length: float
    height: float
    initial_left_segments: int
    initial_bottom_segments: int
    initial_top_segments: int
    lambda_lame: float
    mu: float
    load_x: float
    load_y: float
    regularization: float
    ref_dirichlet: int
    ref_dirichlet_roller: int
    ref_neumann: int
    ref_void: int
    ref_solid: int
    ref_interface: int
    roller_support_required: bool
    mmg_binary: str
    mmg_hmin: float
    mmg_hmax: float
    mmg_hausd: float
    mmg_hgrad: float
    max_iterations: int
    max_line_search: int
    max_coef: float
    min_coef: float
    line_search_increase: float
    line_search_decrease: float
    tolerance: float
    late_tolerance_iteration: int
    late_tolerance: float
    objective_weight: float
    constraint_weight: float
    volume_target: float
    mesh_size_value: float
    eps: float
    normalization_update_iteration: int | None
    advection_backend: str
    supg_tau_scale: float
    topological_derivatives: bool
    topological_interval: int
    topological_last_iteration: int
    topological_volume_fraction: float
    topological_tolerance: float
    topological_volume_tolerance: float
    topological_bisection_iterations: int
    topological_min_distance_from_interface: float
    topological_min_hole_radius: float
    void_cleanup_min_hole_radius: float
    output_every: int
    load_z: float = 0.0
    # how the loop rebuilds phi from the new mesh's tags: "graph" (the signed distance along the mesh's edges, one process
    # only), "exact" (to the interface's facets, gathered whole on every MPI rank) or "auto" (graph on one process, exact
    # on ranks)
    redistancing: str = "auto"
    # the largest constraint step weight alpha_C, in units of the Gauss-Newton step (1 removes the volume error to first
    # order). None keeps sotuto's alpha_C = constraint_weight * h / max|xi_C|, whose step moves the volume by the same
    # amount however small the error is, so it overshoots the target every iteration once it is near. 0.9 is the cap
    # of Feppon's null-space optimizer (nullspace_optimizer's CFL).
    constraint_cap: float | None = None

    @property
    def mesh_size(self) -> float:
        """Representative mesh size used for descent scaling."""

        return self.mesh_size_value

    def line_search_tolerance(self, iteration: int) -> float:
        """Return the iteration-dependent line-search tolerance."""

        if iteration >= self.late_tolerance_iteration:
            return self.late_tolerance
        return self.tolerance

    @property
    def target_volume(self) -> float:
        """Target solid measure for the selected domain."""

        return self.volume_target
