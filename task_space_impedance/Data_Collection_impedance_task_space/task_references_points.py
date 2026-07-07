#!/usr/bin/env python3
"""
Collect reference points for a task and save them to JSON.

- Uses G1RobotArmController (one- or two-layer PD, no impedance needed).
- Each saved point contains:
    * Left and Right EE pose (pos + rot)
    * Full joint dictionary
- JSON structure:

{
  "meta": {
    "created": "...",
    "dds_iface": "enp2s0",
    "dds_domain": 0,
    "mode": "l",
    "controller_dt": 0.02,
    "controller_2l_dt": 0.05
  },
  "points": [
    {
      "id": 0,
      "label": "P0",
      "timestamp": "...",
      "L_ee": {
        "pos": [...],
        "rot": [[...],[...],[...]]
      },
      "R_ee": {
        "pos": [...],
        "rot": [[...],[...],[...]]
      },
      "joints": {
        "LeftShoulderPitch": ...,
        "LeftShoulderRoll": ...,
        ...
      }
    },
    ...
  ]
}
"""

import os
import json
import time
from datetime import datetime

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController

# ----------------------------------------------------------------------
# CONFIG SECTION – EDIT HERE


# ----------------------------------------------------------------------

# DDS / controller config
DDS_DOMAIN   = 0          # 0 = real robot domain, 1 = sim (adjust if needed)
DDS_IFACE    = "enp2s0"   # e.g. "enp2s0" for robot, "lo" for loopback
CTRL_DT      = 0.02
CTRL_2L_DT   = None       # can be None if you want strictly one-layer PD
MODE         = "l"        # "l" for low-level, "h" for high-level topics

# Task / logging config
TASK_FOLDER       = "dance"   # parent folder for this task
POINTS_BASENAME   = "points_run_04" 
    # JSON filename base (without .json)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def np_round_list(arr: np.ndarray, decimals: int = 6):
    """Convert np.array -> rounded python list for JSON."""
    return np.round(np.asarray(arr, dtype=float), decimals=decimals).tolist()


def build_output_path():
    """Return the full path to the output JSON file."""
    base_dir = os.path.dirname(__file__)
    refs_base = os.path.join(base_dir, "refrences_points")
    task_dir = os.path.join(refs_base, TASK_FOLDER)
    os.makedirs(task_dir, exist_ok=True)
    json_path = os.path.join(task_dir, POINTS_BASENAME + ".json")
    return json_path


def build_meta():
    """Build the 'meta' block for JSON."""
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dds_iface": DDS_IFACE,
        "dds_domain": DDS_DOMAIN,
        "mode": MODE,
        "controller_dt": CTRL_DT,
        "controller_2l_dt": CTRL_2L_DT,
    }


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------

def main():
    # 1) DDS init
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS initialized on iface='{DDS_IFACE}', domain={DDS_DOMAIN}.")
    except Exception as e:
        print(f"[WARN] DDS init issue: {e}")

    # 2) Robot controller
    robot = G1RobotArmController(
        ctrl_dt=CTRL_DT,
        ctrl_2l_dt=CTRL_2L_DT,
        mode=MODE,
        visualize=False,
        imp=False     # no impedance in this script
    )

    # Start the joint control loop
    time.sleep(1.0)  # let encoders/DDS settle

    # 3) Prepare output
    out_path = build_output_path()
    meta = build_meta()
    points = []

    print("\n[INFO] Collecting reference points.")
    print(f"[INFO] Output JSON will be saved to:\n       {out_path}\n")
    print("Instructions:")
    print("  - Press ENTER to capture a new point (both arms).")
    print("  - Type a label (or just press ENTER for P0, P1, ...).")
    print("  - Type 'q' and ENTER to quit and save.\n")

    point_id = 0

    try:
        while True:
            cmd = input("Press ENTER to capture, or 'q' to finish: ").strip().lower()
            if cmd == "q":
                print("[INFO] Finishing and saving JSON…")
                break

            # Optional custom label
            default_label = f"P{point_id}"
            label = input(f"  Label for this point [{default_label}]: ").strip()
            if not label:
                label = default_label

            # --- Read state ---
            # 1) joint positions (motor state)
            joints = robot.read_joints()  # dict: {joint_name: pos}
            if not joints:
                print("[WARN] Could not read joints; skipping this point.")
                continue

            # 2) EE poses
            xR, RR = robot.get_R_ee_pose()
            xL, RL = robot.get_L_ee_pose()

            # Build JSON-friendly entry
            point_entry = {
                "id": point_id,
                "label": label,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "R_ee": {
                    "pos": np_round_list(xR, decimals=6),
                    "rot": np_round_list(RR, decimals=6),  # 3x3 as nested list
                },
                "L_ee": {
                    "pos": np_round_list(xL, decimals=6),
                    "rot": np_round_list(RL, decimals=6),
                },
                "joints": {
                    # Ensure standard Python floats and a stable ordering
                    name: float(val) for name, val in joints.items()
                }
            }

            points.append(point_entry)
            print(f"[INFO] Captured point {point_id} with label '{label}'.")
            point_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, saving what was collected…")

    finally:
        # 4) Save JSON
        json_obj = {
            "meta": meta,
            "points": points
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)
            print(f"[INFO] Saved {len(points)} points to:\n       {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON: {e}")

        # 5) Stop robot
        robot.stop()
        print("[INFO] Robot stopped. Done.")


if __name__ == "__main__":
    main()
