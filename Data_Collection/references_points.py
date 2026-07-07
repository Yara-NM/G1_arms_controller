#!/usr/bin/env python3
"""
task_references_points.py

Capture named reference points from the REAL robot (mode='l'):
- Each entry: you type "point_name,index" then press Enter.
- Script records:
    - full joint snapshot (dict of {joint_name: value})
    - Left/Right end-effector 4x4 homogeneous transforms
- Appends to a JSON file under RESULTS_DIR.

Usage:
  python3 task_references_points.py

Notes:
  - Update PKG_PATH and (optionally) the controller import below.
  - DDS iface/domain set in CONFIG.
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
from arms_controller import G1RobotArmController  # <-- change if needed
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# ================= CONFIG =================
# PKG_PATH   = "/home/isr_lab/g1_arm_controller_GPN"  # <- update if needed
RESULTS_DIR= Path(__file__).resolve().parent / "results" / "tasks_reference_points"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON_NAME = "bottle_handover.json"  # final file name under RESULTS_DIR
DDS_DOMAIN = 0                         # 0 for robot, 1 for sim (adjust if needed)
DDS_IFACE  = "enp2s0"                  # e.g., "enp2s0" or "lo"
CTRL_DT    = 0.02
CTRL_2L_DT = 0.05
MODE       = "l"                       # <- Real robot mode

# Gripper settings
GRIPPER_USED = True                    # set False to disable gripper recording
GRIPPER_OPEN = 0                       # open position
GRIPPER_CLOSED = 40                   # closed position
# ==========================================

# Lazy import for gripper controller
DynamixelController = None
if GRIPPER_USED:
    try:
        from gripper_controller import DynamixelController
    except Exception as e:
        print(f"[WARN] Gripper disabled (import failed): {e}")
        GRIPPER_USED = False
        DynamixelController = None


# ---------- helpers ----------
def quat_to_R(q):
    """q = [w, x, y, z] or [x, y, z, w]. Try to guess format; return 3x3 R."""
    q = [float(v) for v in q]
    # heuristic: if first component has the largest magnitude, assume w-first
    if abs(q[0]) >= max(abs(q[1]), abs(q[2]), abs(q[3])):
        w, x, y, z = q
    else:
        x, y, z, w = q
    # normalize
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n > 1e-12:
        w, x, y, z = w/n, x/n, y/n, z/n
    # rotation matrix
    R = np.array([
        [1-2*(y*y+z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),  1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),  2*(y*z + x*w),  1-2*(x*x+y*y)],
    ], dtype=float)
    return R

def to_hmat(pos, rot):
    """
    pos: (3,) xyz
    rot: 3x3 matrix OR quaternion (len==4) OR flat 9-vector. Return 4x4.
    """
    p = np.asarray(pos, dtype=float).reshape(3)
    R = None
    r = np.asarray(rot).reshape(-1)
    if r.size == 9:
        R = r.reshape(3,3).astype(float)
    elif r.size == 4:
        R = quat_to_R(r)
    elif r.size == 3*3:
        R = r.reshape(3,3)
    else:
        # as a last resort, assume already 3x3
        R = np.asarray(rot, dtype=float).reshape(3,3)

    H = np.eye(4, dtype=float)
    H[:3,:3] = R
    H[:3, 3] = p
    return H

def init_dds():
    ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
    print(f"[DDS] iface='{DDS_IFACE}', domain={DDS_DOMAIN} OK")

def get_gripper_states(gripper_hw):
    """Get current gripper states. Returns (left_state, right_state) as binary values (0=open, 1=closed)."""
    if not GRIPPER_USED or gripper_hw is None:
        return 0, 0  # Default to open if grippers not available
    
    try:
        # Try to get actual positions from hardware
        left_pos, right_pos = gripper_hw.get_positions()
        print(f"[DEBUG] Raw gripper positions: L={left_pos}, R={right_pos}")
        # Convert positions to binary states (0=open, 1=closed)
        threshold = (GRIPPER_OPEN + GRIPPER_CLOSED) / 2
        left_state = 1 if left_pos > threshold else 0
        right_state = 1 if right_pos > threshold else 0
        print(f"[DEBUG] Converted states: L={left_state} (threshold={threshold}), R={right_state}")
        return left_state, right_state
    except Exception as e:
        print(f"[WARN] Failed to read gripper states: {e}")
        return 0, 0  # Default to open on error

def main():
    out_path = RESULTS_DIR / OUTPUT_JSON_NAME
    if out_path.exists():
        try:
            data = json.load(open(out_path, "r"))
            assert isinstance(data, dict)
            points = data.get("points", [])
        except Exception:
            data = {"meta": {}, "points": []}
            points = data["points"]
    else:
        data = {"meta": {}, "points": []}
        points = data["points"]

    # Record meta
    data["meta"].update({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dds_iface": DDS_IFACE,
        "dds_domain": DDS_DOMAIN,
        "mode": MODE,
        "controller_dt": CTRL_DT,
        "controller_2l_dt": CTRL_2L_DT,
    })

    # Bring up DDS + controller
    init_dds()
    g1 = G1RobotArmController(
        ctrl_dt=CTRL_DT,
        ctrl_2l_dt=CTRL_2L_DT,
        results_dir=str(RESULTS_DIR),
        mode=MODE,
        visualize=False
    )
    time.sleep(0.3)
    # g1.start()
    time.sleep(0.1)

    # Initialize gripper controller if available
    gripper_hw = None
    if GRIPPER_USED and DynamixelController is not None:
        try:
            gripper_hw = DynamixelController()
            print("[INFO] Gripper controller initialized successfully")
        except Exception as e:
            print(f"[WARN] Failed to initialize gripper controller: {e}")
            gripper_hw = None

    print("\n[READY] Move robot to a pose. Then type entries like:  name,index  (e.g.,  grapes,0)")
    print("        Press ENTER on an empty line to finish.\n")

    while True:
        s = input("Enter point 'name,arm' (or blank to finish): ").strip()
        if not s:
            break
        try:
            name, arm = [x.strip() for x in s.split(",", 1)]
        except ValueError:
            print("  !! Please use format: name,index   (comma-separated)")
            continue

        # Snapshot joints
        jd = g1.read_joints()
        if not isinstance(jd, dict) or not jd:
            print("  !! read_joints() returned nothing; skipping")
            continue

        # EE poses
        try:
            L_pos, L_rot = g1.get_L_ee_pose()
            R_pos, R_rot = g1.get_R_ee_pose()
        except Exception as e:
            print(f"  !! get_*_ee_pose failed: {e}")
            continue

        H_L = to_hmat(L_pos, L_rot).tolist()
        H_R = to_hmat(R_pos, R_rot).tolist()

        # Get gripper states
        left_gripper_state, right_gripper_state = get_gripper_states(gripper_hw)

        entry = {
            "name": name,
            "arm": arm,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joints": {k: float(v) for k, v in jd.items()},
            "left_ee_T_world": H_L,
            "right_ee_T_world": H_R,
            "left_gripper_state": left_gripper_state,
            "right_gripper_state": right_gripper_state
        }
        points.append(entry)
        # Save after each capture
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [+] Saved point '{name},{arm}' with grippers L:{left_gripper_state} R:{right_gripper_state}. Total points: {len(points)}")

    print(f"\n[DONE] Saved {len(points)} reference points -> {out_path}")

if __name__ == "__main__":
    main()
