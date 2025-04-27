#!/usr/bin/env python3
"""
End-to-end regression test & visualiser for *OrientationSpaceExplorer*.

The script can run **one** scene (default: seq-60) or iterate over *all*
files in the `test_scenes` folder.  For every task it:

1.  Loads start/goal poses and the associated occupancy grid.
2.  Executes the planner once to trigger Numba JIT, followed by the
    *timed* run (mean over ``--repeat``).
3.  Saves:
      • the resulting circle-path  →  ``<scene>_ose.txt``
      • a PNG with start/goal, obstacles and final path
4.  Prints a concise runtime report.

No Numba is used here, so feel free to tweak, refactor and profile.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from heurisp import OrientationSpaceExplorer as OSExplorer


# --------------------------------------------------------------------------- #
# Geometry / plotting helpers                                                 #
# --------------------------------------------------------------------------- #
def plot_circle(ax, circle: OSExplorer.CircleNode, *, color="tab:green", alpha=0.6) -> None:
    """Draw *circle* (LCS) as a filled disk plus heading arrow."""
    circ = plt.Circle((circle.x, circle.y), circle.r, color=color, alpha=alpha)
    arrow = plt.Arrow(
        circle.x,
        circle.y,
        0.5 * np.cos(circle.a),
        0.5 * np.sin(circle.a),
        width=0.1,
        color="w",
    )
    ax.add_patch(circ)
    ax.add_patch(arrow)


def plot_grid(ax, grid_map: np.ndarray, grid_res: float) -> None:
    """Render an occupancy grid as red rectangles on *ax*."""
    idx = np.where(grid_map == 255)
    if not idx[0].size:
        return
    row, col = grid_map.shape
    xy2uv = np.array([[0.0, 1.0 / grid_res, row / 2.0],
                      [1.0 / grid_res, 0.0, col / 2.0],
                      [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(xy2uv)
    for u, v in zip(*idx):
        x, y, _ = inv @ np.array([u, v, 1.0])
        ax.add_patch(
            plt.Rectangle(
                (x - grid_res, y - grid_res),
                grid_res,
                grid_res,
                color=(1.0, 0.1, 0.1),
            )
        )


def center2rear(circle: OSExplorer.CircleNode, wheelbase: float = 2.96) -> OSExplorer.CircleNode:
    """Convert circle at *CoM* to rear-axle position (still GCS)."""
    theta, half_wb = circle.a + np.pi, wheelbase / 2.0
    circle.x += half_wb * np.cos(theta)
    circle.y += half_wb * np.sin(theta)
    return circle


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def read_task(scene: Path) -> tuple[OSExplorer.CircleNode, OSExplorer.CircleNode]:
    """Load *_task.txt* → (start, goal) in **GCS**."""
    src, dst = np.loadtxt(scene, delimiter=",")
    start = OSExplorer.CircleNode(x=src[0], y=-src[1], a=-np.radians(src[3]))
    goal = OSExplorer.CircleNode(x=dst[0], y=-dst[1], a=-np.radians(dst[3]))
    return start, goal


def read_grid(scene_png: Path) -> np.ndarray:
    """Load occupancy grid (as uint8)."""
    return cv2.imread(str(scene_png), cv2.IMREAD_UNCHANGED)


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #
def run_scene(filepath: Path, seq: int, *, repeat: int = 1, show: bool = False) -> None:
    """Execute planner for a single *seq* and save artifacts next to it."""
    task_file = filepath / f"{seq}_task.txt"
    grid_file = filepath / f"{seq}_gridmap.png"
    if not task_file.exists() or not grid_file.exists():
        raise FileNotFoundError(f"missing files for scene {seq}")

    # --- Load & convert coordinates ------------------------------------ #
    src_gcs, dst_gcs = read_task(task_file)
    start = center2rear(deepcopy(src_gcs)).gcs2lcs(src_gcs)
    goal = center2rear(deepcopy(dst_gcs)).gcs2lcs(src_gcs)
    grid_ori = deepcopy(src_gcs).gcs2lcs(src_gcs)

    grid_map = read_grid(grid_file)
    grid_res = 0.1

    # --- Planning ------------------------------------------------------- #
    explorer = OSExplorer().initialize(
        start, goal, grid_map=grid_map, grid_res=grid_res, grid_ori=grid_ori
    )

    explorer.exploring()          # warm-up JIT
    t0 = time.perf_counter()
    for _ in range(repeat):
        explorer.exploring()
    dt = (time.perf_counter() - t0) / repeat * 1e3  # ms

    status = "SUCCESS" if explorer.circle_path else "FAIL"
    print(f"[seq {seq:03d}] {status} – {dt:7.2f} ms (mean of {repeat})")

    # --- Save outputs --------------------------------------------------- #
    out_txt = filepath / f"{seq}_ose.txt"
    np.savetxt(out_txt, explorer.path(), delimiter=",", fmt="%.6f")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_aspect("equal")
    ax.set_facecolor((0.2, 0.2, 0.2))
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xticks([])
    ax.set_yticks([])

    plot_grid(ax, explorer.grid_map, grid_res)
    plot_circle(ax, explorer.start, color="tab:blue", alpha=0.8)
    plot_circle(ax, explorer.goal, color="tab:red", alpha=0.8)
    for node in explorer.circle_path:
        plot_circle(ax, node, color="tab:green", alpha=0.3)

    fig.savefig(filepath / f"{seq}_ose.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def iter_sequences(path: Path) -> Iterable[int]:
    """Yield all sequence indices found in *path*."""
    for task in sorted(path.glob("*_task.txt")):
        yield int(task.stem.split("_")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="OrientationSpaceExplorer test-driver")
    parser.add_argument(
        "directory",
        nargs="?",
        default="test_scenes",
        help="folder containing <seq>_task.txt & <seq>_gridmap.png files",
    )
    parser.add_argument("-s", "--seq", type=int, help="run *only* this sequence id")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="timing repetitions")
    parser.add_argument("--show", action="store_true", help="display matplotlib window")
    args = parser.parse_args()

    root = Path(args.directory).expanduser()
    if not root.exists():
        parser.error(f"directory '{root}' does not exist")

    seqs = [args.seq] if args.seq is not None else list(iter_sequences(root))
    if not seqs:
        parser.error("no *_task.txt files found")

    for seq in seqs:
        run_scene(root, seq, repeat=args.repeat, show=args.show)


if __name__ == "__main__":
    main()
