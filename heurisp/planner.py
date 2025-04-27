"""
*Circle-expansion Hybrid A* planner*
===========================================================

This module implements a Hybrid A*-like search operating in **orientation
space**: instead of expanding point states ``(x, y, θ)`` we grow *circles*
``(x, y, θ, r)``, trading branching factor for fewer collision checks.

A **Reeds–Shepp** local planner (``rsplan``) supplies both *g* and *h*
costs, while the inner numeric loops are accelerated with **Numba**.

*Numba stability*: the JIT kernels are extremely picky. Keep them
free of Python containers other than primitives and
``numba.typed.List``, avoid comprehensions and **never** import from
``typing`` inside an ``@njit`` function. All heavyweight logic lives in
regular Python where safety is higher – only tight numeric routines are
jitted.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numba
from numba import njit
import numpy as np
import rsplan

__all__: list[str] = ["path_length", "OrientationSpaceExplorer"]

###############################################################################
# Utility helpers                                                             #
###############################################################################


def path_length(
    q0: Sequence[float],
    q1: Sequence[float],
    radius: float,
    _step: float = 0.1,
) -> float:
    """Return the length of the Reeds–Shepp path between *q0* and *q1*.

    Parameters
    ----------
    q0, q1
        Configurations as ``(x, y, yaw)`` tuples in *global* coordinates.
    radius
        Minimum turning radius for the car-like model.
    _step
        Internal discretisation used by *rsplan*; the default (``0.1``)
        is a good compromise between precision and speed.

    Notes
    -----
    ``rsplan.path`` returns *None* if the two poses cannot be connected
    (obstacles, curvature limits …). In that case we propagate
    :pydata:`math.inf` so the caller can treat it as an inadmissible edge.
    """
    path = rsplan.path(q0, q1, radius, 0, _step)
    return path.total_length if path else math.inf


###############################################################################
# Main planner class                                                          #
###############################################################################


class OrientationSpaceExplorer:
    """Orientation-space Hybrid A* planner.

    Parameters
    ----------
    minimum_radius, maximum_radius
        Bounds for expansion circles.  ``minimum_radius`` acts as a lower
        limit when pruning children; ``maximum_radius`` stabilises the
        search depth.
    minimum_clearance
        Safety distance around obstacles.  Effective circle radius is the
        *geometric* value minus this clearance.
    neighbors
        Angular samples on the half-circle – the actual call creates twice
        that many children (front & back directions).
    maximum_curvature
        Upper bound for the RS local planner; *inverse* of turning radius.
    timeout
        Not used here yet (placeholder for future ROS wrapper).
    overlap_rate
        Two circles are considered *connectable* when they overlap by
        ``overlap_rate × min(r_small, r_big)``.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        minimum_radius: float = 0.20,
        maximum_radius: float = 2.96,
        minimum_clearance: float = 1.02,
        neighbors: int = 32,
        maximum_curvature: float = 0.2,
        timeout: float = 1.0,
        overlap_rate: float = 0.5,
    ) -> None:
        # Tunables
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.minimum_clearance = minimum_clearance
        self.neighbors = neighbors
        self.maximum_curvature = maximum_curvature
        self.timeout = timeout
        self.overlap_rate = overlap_rate

        # Planning state (populated in `initialize`)
        self.start: Optional[OrientationSpaceExplorer.CircleNode] = None
        self.goal: Optional[OrientationSpaceExplorer.CircleNode] = None
        self.grid_ori: Optional[OrientationSpaceExplorer.CircleNode] = None
        self.grid_map: Optional[np.ndarray] = None
        self.grid_res: Optional[float] = None
        self.grid_pad: Optional[np.ndarray] = None
        self.obstacle: int = 255  # occupancy threshold

    # ------------------------------------------------------------------ #
    # Public façade                                                      #
    # ------------------------------------------------------------------ #

    def exploring(self, plotter=None) -> bool:
        """Run the A* loop until *goal* is reached or OPEN becomes empty.

        The OPEN set is implemented as a *descending* list so we can pop
        from the tail in *O(1)*.  This is mildly sub-optimal compared with a
        binary heap but much friendlier for Numba inter-op.

        Returns
        -------
        bool
            *True* if a path was found, *False* otherwise.
        """
        # ---------------- initialisation ---------------- #
        close_set = numba.typed.List()  # stores (x, y, a, r)
        open_set: List[Tuple[float, OrientationSpaceExplorer.CircleNode]] = [
            (self.start.f, self.start)  # type: ignore[index]
        ]
        # Seed a dummy element to give Numba a concrete type
        close_set.append((0.0, 0.0, 0.0, 0.0))
        close_set.pop()

        # ---------------------- main loop ---------------------- #
        while open_set:
            circle = self.pop_top(open_set)  # smallest *f*-value

            # Early-exit heuristic: once the cheapest node in OPEN is more
            # expensive than the best goal-cost we discovered, the
            # current path is optimal (A* property).
            if self.goal.f < circle.f:  # type: ignore[operator]
                return True

            if not self.exist(circle, close_set):
                expansion = self.expand(circle)
                self.merge(expansion, open_set)

                # Greedy direct connection — if circles overlap we can try
                # jumping straight to GOAL without further expansions.
                if self.overlap(circle, self.goal) and circle.f < self.goal.g:
                    self.goal.g = circle.f
                    self.goal.f = self.goal.g + self.goal.h
                    self.goal.set_parent(circle)

                # Mark state as explored (CLOSED)
                close_set.append((circle.x, circle.y, circle.a, circle.r))

            if plotter:  # optional live-view hook
                plotter([circle])

        return False  # OPEN exhausted → failure

    # ---------------- convenience getters ---------------- #

    @property
    def circle_path(self) -> List["OrientationSpaceExplorer.CircleNode"]:
        """Reconstruct solution by chasing parents back to START."""
        if self.goal is None or self.goal.parent is None:
            return []
        path, parent = [self.goal], self.goal.parent
        while parent:
            path.append(parent)
            parent = parent.parent
        path.reverse()
        return path

    def path(self) -> List[Tuple[float, float, float, float]]:
        """Return path as raw tuples for external consumers (ROS, JSON…)."""
        return [(p.x, p.y, p.a, p.r) for p in self.circle_path]

    # ------------------------------------------------------------------ #
    # Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        start: "OrientationSpaceExplorer.CircleNode",
        goal: "OrientationSpaceExplorer.CircleNode",
        grid_map: np.ndarray,
        grid_res: float,
        grid_ori: "OrientationSpaceExplorer.CircleNode",
        obstacle: int = 255,
    ) -> "OrientationSpaceExplorer":
        """Attach map data and compute static lookup tables.

        ``grid_map``  – square occupancy raster centred at ``grid_ori``.  
        All geometry is expressed in **world metres** while indices inside
        Numba kernels use **grid cells** – conversion happens via
        ``grid_res``.
        """
        self.start, self.goal = start, goal
        self.grid_map, self.grid_res, self.grid_ori, self.obstacle = (
            grid_map,
            grid_res,
            grid_ori,
            obstacle,
        )

        # Build a padded map so clearance queries never run out-of-bounds.
        pad_pix = int(
            math.ceil((self.maximum_radius + self.minimum_clearance) / self.grid_res)
        )
        self.grid_pad = np.pad(
            self.grid_map,
            ((pad_pix, pad_pix), (pad_pix, pad_pix)),
            "constant",
            constant_values=(
                (self.obstacle, self.obstacle),
                (self.obstacle, self.obstacle),
            ),
        )

        # ------------------ seed heuristic ------------------ #
        self.start.r = self.clearance(self.start) - self.minimum_clearance
        self.start.g = 0.0
        self.start.h = path_length(
            (self.start.x, self.start.y, self.start.a),
            (self.goal.x, self.goal.y, self.goal.a),
            1.0 / self.maximum_curvature,
        )
        self.start.f = self.start.h  # g==0

        self.goal.r = self.clearance(self.goal) - self.minimum_clearance
        self.goal.g = math.inf  # to be discovered
        self.goal.h = 0.0
        self.goal.f = math.inf

        return self

    # ------------------------------------------------------------------ #
    # Open / Closed set helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def merge(
        expansion: Sequence["OrientationSpaceExplorer.CircleNode"],
        open_set: List[Tuple[float, "OrientationSpaceExplorer.CircleNode"]],
    ) -> None:
        """Insert nodes from *expansion* into OPEN and keep it sorted."""
        open_set.extend([(x.f, x) for x in expansion])
        open_set.sort(reverse=True)  # cheap for small OPEN (< few k)

    @staticmethod
    def pop_top(
        open_set: List[Tuple[float, "OrientationSpaceExplorer.CircleNode"]]
    ) -> "OrientationSpaceExplorer.CircleNode":
        """Remove and return the node with the *lowest* ``f`` cost."""
        return open_set.pop()[-1]

    # ------------------------------------------------------------------ #
    # Numba-accelerated helpers                                          #
    # ------------------------------------------------------------------ #

    def exist(self, circle: "OrientationSpaceExplorer.CircleNode", close_set) -> bool:
        """Check duplicate in CLOSED using a relaxed SE(2) metric."""
        state = (circle.x, circle.y, circle.a)
        return self.jit_exist(state, close_set, self.maximum_curvature)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def jit_exist(state, close_set, maximum_curvature):
        """Return *True* if *state* is already represented in CLOSED."""
        def metric(p, q, curv):
            euclid = math.hypot(p[0] - q[0], p[1] - q[1])
            ang = abs(p[2] - q[2])
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            rs = ang / curv  # convert angular delta to path length
            return euclid if euclid > rs else rs

        for item in close_set:
            # item[-1] is the circle radius stored alongside (x,y,a)
            if metric(state, item, maximum_curvature) < item[-1] - 0.1:
                return True
        return False

    # ----------------------------- goal connect ------------------------- #

    def overlap(
        self,
        circle: "OrientationSpaceExplorer.CircleNode",
        goal: "OrientationSpaceExplorer.CircleNode",
    ) -> bool:
        """Geometric test whether *circle* overlaps *goal* sufficiently."""
        return self.jit_overlap(
            (circle.x, circle.y, circle.r),
            (goal.x, goal.y, goal.r),
            self.overlap_rate,
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def jit_overlap(circle, goal, rate):
        euclid = math.hypot(circle[0] - goal[0], circle[1] - goal[1])
        r_small, r_big = (
            (circle[2], goal[2]) if circle[2] < goal[2] else (goal[2], circle[2])
        )
        return euclid < r_small * rate + r_big

    # ------------------------------------------------------------------ #
    # Expansion                                                          #
    # ------------------------------------------------------------------ #

    def expand(self, circle: "OrientationSpaceExplorer.CircleNode") -> List[
        "OrientationSpaceExplorer.CircleNode"
    ]:
        """Generate child circles around *circle* and wrap them as nodes."""

        # Child post-processing – enrich raw `(x,y,θ,r)` tuples with costs.
        def _wrap(state):
            child = OrientationSpaceExplorer.CircleNode(*state)
            child.set_parent(circle)

            # Heuristic cost-to-go (RS distance to GOAL)
            child.h = path_length(
                (child.x, child.y, child.a),
                (self.goal.x, self.goal.y, self.goal.a),
                1.0 / self.maximum_curvature,
            )

            # Exact cost-from-come (RS distance parent → child)
            child.g = circle.g + path_length(
                (circle.x, circle.y, circle.a),
                (child.x, child.y, child.a),
                1.0 / self.maximum_curvature,
            )
            child.f = child.g + child.h
            return child

        neighbors = self.jit_neighbors(
            (circle.x, circle.y, circle.a), circle.r, self.neighbors
        )
        children = self.jit_children(
            neighbors,
            (self.grid_ori.x, self.grid_ori.y, self.grid_ori.a),
            self.grid_pad,
            self.grid_map,
            self.grid_res,
            self.maximum_radius,
            self.minimum_radius,
            self.minimum_clearance,
            self.obstacle,
        )
        return list(map(_wrap, children))

    # -------------------------- jit kernels ---------------------------- #

    @staticmethod
    @njit(fastmath=True, cache=True)
    def jit_children(
        neighbors,
        origin,
        grid_pad,
        grid_map,
        grid_res,
        maximum_radius,
        minimum_radius,
        minimum_clearance,
        obstacle,
    ):
        """Filter *neighbors* by clearance and annotate with admissible radius."""

        def clearance(state):
            # Transform *state* into *grid* frame centred at ORIGIN ---------
            s_x, s_y, s_a = origin
            c_x, c_y, _ = state
            x = (c_x - s_x) * math.cos(s_a) + (c_y - s_y) * math.sin(s_a)
            y = -(c_x - s_x) * math.sin(s_a) + (c_y - s_y) * math.cos(s_a)
            u = int(math.floor(y / grid_res + grid_map.shape[0] / 2))
            v = int(math.floor(x / grid_res + grid_map.shape[0] / 2))

            # Extract square patch around query point (padding ensures bounds)
            size = int(math.ceil((maximum_radius + minimum_clearance) / grid_res))
            sub = grid_pad[u : u + 2 * size + 1, v : v + 2 * size + 1]

            # Distance transform (simplified): find nearest obstacle cell
            rows, cols = np.where(sub >= obstacle)
            if len(rows):
                row = np.fabs(rows - size) - 1
                col = np.fabs(cols - size) - 1
                return (np.sqrt(row**2 + col**2) * grid_res).min()
            return size * grid_res  # no obstacles in sight

        children = numba.typed.List()
        children.append((0.0, 0.0, 0.0, 0.0))
        children.pop()

        for n in neighbors:
            r = min(clearance(n) - minimum_clearance, maximum_radius)
            if r > minimum_radius:
                children.append((n[0], n[1], n[2], r))
        return children

    @staticmethod
    @njit(fastmath=True, cache=True)
    def jit_neighbors(state, radius, number):
        """Generate *number* evenly spaced points on a circle around *state*."""

        def lcs2gcs(p):
            x, y, a = p
            xo, yo, ao = state
            x1 = x * math.cos(ao) - y * math.sin(ao) + xo
            y1 = x * math.sin(ao) + y * math.cos(ao) + yo
            return x1, y1, a + ao

        neigh = numba.typed.List()
        neigh.append((0.0, 0.0, 0.0))
        neigh.pop()

        for n in np.radians(np.linspace(-90.0, 90.0, number // 2)):
            p = (radius * math.cos(n), radius * math.sin(n), n)
            o = (radius * math.cos(n + math.pi), radius * math.sin(n + math.pi), n)
            neigh.append(lcs2gcs(p))
            neigh.append(lcs2gcs(o))
        return neigh

    # ------------------------------------------------------------------ #
    # Clearance helper                                                  #
    # ------------------------------------------------------------------ #

    def clearance(self, circle: "OrientationSpaceExplorer.CircleNode") -> float:
        origin = (self.grid_ori.x, self.grid_ori.y, self.grid_ori.a)
        coord = (circle.x, circle.y, circle.a)
        return self.jit_clearance(
            coord,
            origin,
            self.grid_pad,
            self.grid_map,
            self.grid_res,
            self.maximum_radius,
            self.minimum_clearance,
            self.obstacle,
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def jit_clearance(
        coord,
        origin,
        grid_pad,
        grid_map,
        grid_res,
        maximum_radius,
        minimum_clearance,
        obstacle,
    ):
        s_x, s_y, s_a = origin
        c_x, c_y, _ = coord
        x = (c_x - s_x) * math.cos(s_a) + (c_y - s_y) * math.sin(s_a)
        y = -(c_x - s_x) * math.sin(s_a) + (c_y - s_y) * math.cos(s_a)
        u = int(math.floor(y / grid_res + grid_map.shape[0] / 2))
        v = int(math.floor(x / grid_res + grid_map.shape[0] / 2))
        size = int(math.ceil((maximum_radius + minimum_clearance) / grid_res))
        sub = grid_pad[u : u + 2 * size + 1, v : v + 2 * size + 1]
        rows, cols = np.where(sub >= obstacle)
        if len(rows):
            row = np.fabs(rows - size) - 1
            col = np.fabs(cols - size) - 1
            return (np.sqrt(row**2 + col**2) * grid_res).min()
        return size * grid_res

    # ------------------------------------------------------------------ #
    # Circle node                                                        #
    # ------------------------------------------------------------------ #

    circle_node = numba.deferred_type()

    spec = [
        ("x", numba.float64),
        ("y", numba.float64),
        ("a", numba.float64),
        ("r", numba.optional(numba.float64)),
        ("h", numba.float64),
        ("g", numba.float64),
        ("f", numba.float64),
        ("parent", numba.optional(circle_node)),
        ("children", numba.optional(numba.types.List(circle_node))),
    ]

    # For numba ≥0.59 you can uncomment the next line.
    # @numba.jitclass(spec)
    class CircleNode:
        """Lightweight container for a circle-state in SE(2)."""

        __slots__ = (
            "x",
            "y",
            "a",
            "r",
            "h",
            "g",
            "f",
            "parent",
            "children",
        )

        def __init__(
            self,
            x: float | None = None,
            y: float | None = None,
            a: float | None = None,
            r: float | None = None,
        ) -> None:
            self.x = float("nan") if x is None else float(x)
            self.y = float("nan") if y is None else float(y)
            self.a = float("nan") if a is None else float(a)
            self.r = r
            self.h = math.inf  # heuristic cost-to-go
            self.g = math.inf  # actual cost-from-come
            self.f = math.inf  # = g + h
            self.parent: Optional[
                "OrientationSpaceExplorer.CircleNode"
            ] = None  # back-pointer
            self.children = None  # unused for now

        # --------------------- helpers ---------------------- #
        def set_parent(self, circle: "OrientationSpaceExplorer.CircleNode") -> None:
            self.parent = circle

        def lcs2gcs(
            self, circle: "OrientationSpaceExplorer.CircleNode"
        ) -> "OrientationSpaceExplorer.CircleNode":
            """Transform *self* from LCS of *circle* to **global** CS."""
            xo, yo, ao = circle.x, circle.y, circle.a
            x = self.x * math.cos(ao) - self.y * math.sin(ao) + xo
            y = self.x * math.sin(ao) + self.y * math.cos(ao) + yo
            self.x, self.y, self.a = x, y, self.a + ao
            return self

        def gcs2lcs(
            self, circle: "OrientationSpaceExplorer.CircleNode"
        ) -> "OrientationSpaceExplorer.CircleNode":
            """Transform *self* from global CS to LCS of *circle*."""
            xo, yo, ao = circle.x, circle.y, circle.a
            x = (self.x - xo) * math.cos(ao) + (self.y - yo) * math.sin(ao)
            y = -(self.x - xo) * math.sin(ao) + (self.y - yo) * math.cos(ao)
            self.x, self.y, self.a = x, y, self.a - ao
            return self

        # Python’s `heapq` uses `<` for priority comparison
        def __lt__(self, other: "OrientationSpaceExplorer.CircleNode") -> bool:
            return self.g < other.g


# For numba ≥0.59 you can uncomment the next lines
# Finalise deferred numba type – keep *after* class definition
# OrientationSpaceExplorer.circle_node.define(
#     OrientationSpaceExplorer.CircleNode.class_type.instance_type  # type: ignore[attr-defined]
# )