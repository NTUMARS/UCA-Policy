#!/usr/bin/env python3
"""Align new real-robot zarr EE orientations to the old ur5_pick frame.

Only action/state rotation-6D (dims 3:9) are transformed:
    R_aligned = R_new @ R_delta
where R_delta is estimated from mean orientations:
    R_delta = mean_R(new_pick)^T @ mean_R(old_pick)

xyz, gripper, point_cloud, episode_ends are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr


def rot6_to_R(rot6: np.ndarray) -> np.ndarray:
    c1 = rot6[:, :3]
    c2 = rot6[:, 3:6]
    b1 = c1 / np.clip(np.linalg.norm(c1, axis=1, keepdims=True), 1e-12, None)
    b2 = c2 - np.sum(b1 * c2, axis=1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=1, keepdims=True), 1e-12, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def R_to_rot6(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    M = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def project_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    det = np.linalg.det(Rp)
    if np.ndim(det) == 0:
        if det < 0:
            U = U.copy()
            U[:, -1] *= -1
            Rp = U @ Vt
        return Rp
    neg = det < 0
    if np.any(neg):
        U = U.copy()
        U[neg, :, -1] *= -1
        Rp = U @ Vt
    return Rp


def align_pose10(x: np.ndarray, R_delta: np.ndarray) -> np.ndarray:
    out = x.copy()
    R = rot6_to_R(x[:, 3:9])
    R_aligned = project_so3(R @ R_delta[None, ...])
    out[:, 3:9] = R_to_rot6(R_aligned)
    return out


def export_aligned(src_path: Path, dst_path: Path, R_delta: np.ndarray, meta: dict):
    if dst_path.exists():
        shutil.rmtree(dst_path)
    # full copy keeps point_cloud chunks/compressor identical
    shutil.copytree(src_path, dst_path)

    root = zarr.open(str(dst_path), mode="r+")
    action = np.asarray(root["data"]["action"][:])
    state = np.asarray(root["data"]["state"][:])
    action_a = align_pose10(action, R_delta)
    state_a = align_pose10(state, R_delta)
    root["data"]["action"][:] = action_a
    root["data"]["state"][:] = state_a

    meta_path = Path(str(dst_path) + ".align.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    ends = np.asarray(root["meta"]["episode_ends"][:])
    keys = list(root["data"].keys())
    Ra = rot6_to_R(action_a[:, 3:9])
    print(f"[ok] {src_path.name} -> {dst_path.name}")
    print(f"     data keys={keys}")
    print(f"     steps={action_a.shape[0]} episodes={len(ends)}")
    print(f"     aligned rot mean c1={action_a[:,3:6].mean(0)}")
    print(f"     aligned rot mean c2={action_a[:,6:9].mean(0)}")
    print(f"     det(R) mean={np.linalg.det(Ra).mean():.6f}")
    assert "point_cloud" in keys, "point_cloud missing after copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "UCA" / "data",
    )
    args = ap.parse_args()
    root: Path = args.data_root

    old_pick = root / "ur5_pick_dataset.zarr"
    new_pick = root / "ur5_pick_rebuttal_dataset.zarr"
    new_stack = root / "ur5_stack_qube_dataset.zarr"
    out_pick = root / "ur5_pick_rebuttal_aligned_dataset.zarr"
    out_stack = root / "ur5_stack_qube_aligned_dataset.zarr"

    old_act = np.asarray(zarr.open(str(old_pick), "r")["data"]["action"][:])
    new_act = np.asarray(zarr.open(str(new_pick), "r")["data"]["action"][:])
    R_old = mean_rotation(rot6_to_R(old_act[:, 3:9]))
    R_new = mean_rotation(rot6_to_R(new_act[:, 3:9]))
    R_delta = project_so3(R_new.T @ R_old)

    from scipy.spatial.transform import Rotation

    euler = Rotation.from_matrix(R_delta).as_euler("xyz", degrees=True).tolist()
    meta = {
        "method": "R_aligned = R_new @ R_delta, R_delta = mean_R(new_pick).T @ mean_R(old_pick)",
        "reference_old": str(old_pick),
        "reference_new_for_delta": str(new_pick),
        "R_delta": R_delta.tolist(),
        "R_delta_euler_xyz_deg": euler,
        "note": "Only action/state dims[3:9] (6D rotation) are transformed; xyz/gripper/point_cloud unchanged.",
    }
    print("R_delta euler xyz deg:", euler)

    export_aligned(new_pick, out_pick, R_delta, {**meta, "source": str(new_pick)})
    export_aligned(new_stack, out_stack, R_delta, {**meta, "source": str(new_stack)})
    print("old pick mean c1/c2:", old_act[:, 3:6].mean(0), old_act[:, 6:9].mean(0))


if __name__ == "__main__":
    main()
