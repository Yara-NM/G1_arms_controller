#!/usr/bin/env python3
"""
collect_data.py

Data-collection script for the G1 task-space spring-damper pipeline.

Architecture (4 threads):
  - control_thread: SpringDamperCartesian -> IK -> joint PD at CTRL_DT
  - task_thread   : updates spring targets (zero + waypoints), checks "reached"
  - state_logger  : logs t, EE_meas (R/L), EE_ref (R/L), joints
  - video_thread  : logs RGB video + frame timestamps

Outputs:
  - database/<TASK_NAME>/<RUN_NAME>/meta.json
  - database/<TASK_NAME>/<RUN_NAME>/task_space_log.json
  - database/<TASK_NAME>/<RUN_NAME>/camera.mp4
  - database/<TASK_NAME>/<RUN_NAME>/frame_timestamps.csv
"""

import os
import json
import time
import threading
from datetime import datetime
import cv2
import csv
from pathlib import Path
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian

# -----------------------------------------------------------------------------
# Gripper (currently OFF)
# -----------------------------------------------------------------------------
GRIPPER_USED = True
DynamixelController = None
if GRIPPER_USED:
    try:
        from gripper_controller import DynamixelController
    except Exception as e:
        print(f"[WARN] Gripper disabled (import failed): {e}")
        GRIPPER_USED = False
        DynamixelController = None

# Commanded angles (edit to your calibration)
LEFT_GRIPPER_OPEN   = 0
LEFT_GRIPPER_CLOSED = 200
# 65 for soft balls
# 60 for eggs
# 80 for soya sauce 
# toy star 130
# plant 180
# cloth 200
RIGHT_GRIPPER_OPEN   = 0
RIGHT_GRIPPER_CLOSED = 110

DEFAULT_LEFT_GRIPPER  = 0
DEFAULT_RIGHT_GRIPPER = 0
# -----------------------------------------------------------------------------
# 1) Global config 
# -----------------------------------------------------------------------------
# ---- Results saving config ----
TASK_NAME = "Demo"    # e.g. "bottle_handover"
RUN_NAME  = time.strftime("run_%Y%m%d_%H%M%S")   

TASK_DESCRIPTION = "simulation"
# ---- Reference points source ----
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_JSON_PATH = ROOT / "Data_Collection_impedance_task_space" / "refrences_points" / "grasping" / "points_run_01.json"
# REFERENCE_JSON_PATH = "/home/isr_lab/g1/task_space_impedance/Data_Collection_impedance_task_space/refrences_points/follow surface/points_run_02.json"

# ---- DDS / controller config ----
DDS_DOMAIN = 0          # 0 = real robot, 1 = sim (adjust to your setup)
DDS_IFACE  = "enp2s0"   # e.g. "enp2s0" for real robot, "lo" for loopback
CTRL_DT    = 0.02       # 50 Hz joint + spring–damper loop
CTRL_2L_DT = None       # Force one-layer PD controller
MODE       = "l"        # "l" for low-level

# ---- Spring–damper gains ----
# Base gains (you can tune these)
BASE_SPRING_K_P = (15.0, 15.0, 12.0) 
BASE_SPRING_D_P = (6.0,  6.0,  5.0)
BASE_SPRING_M_P = (1.5,  1.5,  1.5)
BASE_SPRING_K_R = (3.0,  3.0,  2.0)
BASE_SPRING_D_R = (1.0,  1.0,  0.8)
BASE_SPRING_M_R = (0.2,  0.2,  0.2)

# follow thw surface K=  6,6,3  D =  2,2,2
# grasping k= 5, 5, 4, D = 2 , 2 , 1.5  

# NOTE: Separate per-arm gains (you can make L/R different if you want)
RIGHT_SPRING_K_P = (5.0, 5.0, 4.0) # BASE_SPRING_K_P
RIGHT_SPRING_D_P = (2.0,  2.0,  1.5)  # BASE_SPRING_D_P
RIGHT_SPRING_M_P = (1.5,  1.5,  1.5)  # BASE_SPRING_M_P
RIGHT_SPRING_K_R = (3.0,  3.0,  2.0)  # BASE_SPRING_K_R
RIGHT_SPRING_D_R = (1.0,  1.0,  0.8)  # BASE_SPRING_D_R
RIGHT_SPRING_M_R = (0.2,  0.2,  0.2)  # BASE_SPRING_M_R

LEFT_SPRING_K_P  = (5.0, 5.0, 4.0) # BASE_SPRING_K_P
LEFT_SPRING_D_P = (2.0,  2.0,  1.5)  # BASE_SPRING_D_P
LEFT_SPRING_M_P = (1.5,  1.5,  1.5)  # BASE_SPRING_M_P
LEFT_SPRING_K_R = (3.0,  3.0,  2.0)  # BASE_SPRING_K_R
LEFT_SPRING_D_R = (1.0,  1.0,  0.8)  # BASE_SPRING_D_R
LEFT_SPRING_M_R = (0.2,  0.2,  0.2)  # BASE_SPRING_M_R

# ---- Camera ----
CAM_INDEX   = 6
CAM_RES     = (1280, 720)
CAM_FPS_REQ = 30
VIDEO_FPS   = 30     # encode target; will run as fast as camera allows

# ---- Zero pose (same style as in test_interactive_task_space.py) ----
# You can edit these if your zero pose changes.
ZERO_RIGHT_POS = np.array([0.32686 - 0.10, -0.15184 - 0.10, 0.07149])
ZERO_RIGHT_ROT = np.array([[ 0.99366,  0.01195,  0.11179],
                           [-0.01149,  0.99992, -0.00482],
                           [-0.11184,  0.00350,  0.99372]])

ZERO_LEFT_POS  = np.array([0.32683 - 0.10,  0.15156 + 0.10, 0.07145])
ZERO_LEFT_ROT  = np.array([[ 0.99365, -0.01188,  0.11188],
                           [ 0.01129,  0.99992,  0.00597],
                           [-0.11195, -0.00467,  0.99370]])

# ---- Task execution thresholds ----
LOG_DT = 0.02  # [s] logger period (can be == CTRL_DT)



# -----------------------------------------------------------------------------
# 2) Helpers
# -----------------------------------------------------------------------------

def np_round_list(arr, decimals=6):
    """Convert np.array -> rounded python list for JSON."""
    return np.round(np.asarray(arr, dtype=float), decimals=decimals).tolist()

def load_reference_points(path):
    """
    Load reference EE poses from a JSON file created by task_references_points.py.
    Expected structure:
    {
      "meta": {...},
      "points": [
        {
          "id": 0,
          "label": "P0",
          "timestamp": "...",
          "R_ee": { "pos": [...], "rot": [[...],[...],[...]] },
          "L_ee": { "pos": [...], "rot": [[...],[...],[...]] },
          "joints": { ... }
        },
        ...
      ]
    }

    Internally we normalize everything to:
      ref = {
        "name": <label or fallback>,
        "right": {"x": np.array, "R": np.array(3x3)},
        "left":  {"x": np.array, "R": np.array(3x3)},
        "joints": {...},
        "t": <optional timestamp>
      }
    """
    with open(path, "r") as f:
        data = json.load(f)

    pts = data.get("points", [])
    out = []

    for p in pts:

        R_ee = p["R_ee"]
        L_ee = p["L_ee"]

        try:
            xR = np.asarray(R_ee["pos"], dtype=float).reshape(3)
            RR = np.asarray(R_ee["rot"], dtype=float).reshape(3, 3)
            xL = np.asarray(L_ee["pos"], dtype=float).reshape(3)
            RL = np.asarray(L_ee["rot"], dtype=float).reshape(3, 3)

        except Exception as e:
            raise ValueError(f"Reference point is missing R_ee.pos or R_ee.rot")


        out.append({
            "name":   p.get("label", p.get("name", "")),
            "right":  {"x": xR, "R": RR},
            "left":   {"x": xL, "R": RL},
            "joints": p.get("joints", {}),
            "t":      p.get("timestamp", p.get("t", None)),
        })

    return out



def open_camera(idx, res, fps_req):

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    if fps_req:
        cap.set(cv2.CAP_PROP_FPS, fps_req)
    time.sleep(0.4)
    act_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    act_fps= cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] Opened index={idx} at {act_w}x{act_h} @ {act_fps:.1f} fps")

    return cap, (act_w, act_h), float(act_fps or 0.0)

def next_run_dir(results_root: str, run_name: str) -> Path:
   
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    cand = root / run_name
    cand.mkdir(parents=True, exist_ok=False)
    return cand


# --- Reference helpers --------------------------------------------------------
def get_ref_by_index(refs, idx):
    return refs[idx]

def get_ref_by_name(refs, name_substr):
    name_substr = name_substr.lower()
    for r in refs:
        if name_substr in r["name"].lower():
            return r
    return None

def ref_right_pose(ref):  # -> (xR, RR)
    return ref["right"]["x"], ref["right"]["R"]

def ref_left_pose(ref):   # -> (xL, RL)
    return ref["left"]["x"], ref["left"]["R"]


# -----------------------------------------------------------------------------
# 3) Threads
# -----------------------------------------------------------------------------

# Shared state between threads
state_lock = threading.Lock()
shared_state = {
    "ref_xR": ZERO_RIGHT_POS.copy(),
    "ref_RR": ZERO_RIGHT_ROT.copy(),
    "ref_xL": ZERO_LEFT_POS.copy(),
    "ref_RL": ZERO_LEFT_ROT.copy(),
}

log_data = []           # list of samples (built by logger thread)
stop_event = threading.Event()

# --- 3.1 Control loop thread (spring + IK + PD) -------------------------------

def control_thread(robot, spring_R, spring_L):
    """
    Runs at CTRL_DT:
      - step spring-damper in task space
      - send EE pose to robot IK+PD
      - publish commanded EE pose into shared_state
    """
    print("[CTRL] Control thread started.")

    t0 = time.time()
    while not stop_event.is_set():
        t = time.time() - t0

        # Spring–damper integration
        xR_d, RR_d, _ = spring_R.step(CTRL_DT)
        xL_d, RL_d, _ = spring_L.step(CTRL_DT)

        # Send EE commands via IK to joint PD
        robot.move_arms_with_Rt(
            left_R=RL_d,  left_t=xL_d,
            right_R=RR_d, right_t=xR_d
        )

        # Publish commanded pose for logger
        with state_lock:
            shared_state["ref_xR"] = xR_d.copy()
            shared_state["ref_RR"] = RR_d.copy()
            shared_state["ref_xL"] = xL_d.copy()
            shared_state["ref_RL"] = RL_d.copy()

        time.sleep(CTRL_DT)

    print("[CTRL] Control thread exiting.")

# --- 3.2 Task thread (high-level go_to) --------------------------------------

def task_thread(robot, spring_R, spring_L, waypoints, gripper):
    """
    High-level task logic. It assumes control_thread is running and always
    tracking the spring internal command.

    We only:
      - set spring targets (zero, or waypoint poses)
      - monitor measured EE error until within tol
      - hold on target for some time
    """
    def do_gripper_action(name, gripper_obj):
        """
        Simple naming-based script for grippers.
        You can change this mapping to whatever convention you want.
        """
        if gripper_obj is None or not GRIPPER_USED:
            return

        lname = name.lower()

        # Examples — edit to your task:
        #  - "r_close"  -> close right gripper
        #  - "r_open"   -> open right
        #  - "l_close"  -> close left
        #  - "l_open"   -> open left

        try:
            if "r_close" in lname:
                gripper_obj.set_right_gripper(RIGHT_GRIPPER_CLOSED)
                print("[TASK] Right gripper: CLOSE")
            elif "r_open" in lname:
                gripper_obj.set_right_gripper(RIGHT_GRIPPER_OPEN)
                print("[TASK] Right gripper: OPEN")

            if "l_close" in lname:
                gripper_obj.set_left_gripper(LEFT_GRIPPER_CLOSED)
                print("[TASK] Left gripper: CLOSE")
            elif "l_open" in lname:
                gripper_obj.set_left_gripper(LEFT_GRIPPER_OPEN)
                print("[TASK] Left gripper: OPEN")
        except Exception as e:
            print(f"[WARN] Gripper command failed for '{name}': {e}")

    def go_to_zero(tol=0.01, hold=0.5, timeout=20.0):
        """Go to the ZERO pose using spring targets."""
        print(f"[TASK] go_to('zero', tol={tol}, hold={hold})")

        # Set asymptotic targets
        spring_R.set_target(ZERO_RIGHT_POS, ZERO_RIGHT_ROT)
        spring_L.set_target(ZERO_LEFT_POS,  ZERO_LEFT_ROT)

        start_time = time.time()
        reached = False
        reached_since = None

        while not stop_event.is_set():
            t = time.time() - start_time

            # Measure actual EE
            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()

            err_R = np.linalg.norm(ZERO_RIGHT_POS - xR_meas)
            err_L = np.linalg.norm(ZERO_LEFT_POS  - xL_meas)
            err   = max(err_R, err_L)

            if t < 2.0 or int(t) % 1 == 0:
                # light logging
                # (not every cycle, but it's fine to keep as-is)
                pass

            if err < tol:
                if not reached:
                    reached = True
                    reached_since = time.time()
                    print(f"[TASK] Zero reached (err={err:.4f} m). Holding...")
                else:
                    if (time.time() - reached_since) > hold:
                        print("[TASK] Finished holding zero.")
                        return True
            else:
                reached = False
                reached_since = None

            if t > timeout:
                print(f"[WARN] go_to_zero timeout (err={err:.4f} m).")
                return False

            time.sleep(CTRL_DT)

    def go_to_ref(ref, tol=0.015, hold=0.5, timeout=20.0):
        """
        Go to a reference waypoint. Uses full dual-arm target by default.
        """
        name = ref.get("name", "<unnamed>")
        xR_ref, RR_ref = ref_right_pose(ref)
        xL_ref, RL_ref = ref_left_pose(ref)

        print(f"[TASK] go_to('{name}', tol={tol}, hold={hold})")

        spring_R.set_target(xR_ref, RR_ref)
        spring_L.set_target(xL_ref, RL_ref)

        start_time = time.time()
        reached = False
        reached_since = None

        while not stop_event.is_set():
            t = time.time() - start_time

            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()

            err_R = np.linalg.norm(xR_ref - xR_meas)
            err_L = np.linalg.norm(xL_ref - xL_meas)
            err   = max(err_R, err_L)

            if err < tol:
                if not reached:
                    reached = True
                    reached_since = time.time()
                    print(f"[TASK] '{name}' reached (err={err:.4f} m). Holding...")
                else:
                    if (time.time() - reached_since) > hold:
                        print(f"[TASK] Finished holding '{name}'.")
                        return True
            else:
                reached = False
                reached_since = None

            if t > timeout:
                print(f"[WARN] go_to('{name}') timeout (err={err:.4f} m).")
                return False

            time.sleep(CTRL_DT)

    def go_to_ref_right_only(ref, tol=0.015, hold=0.5, timeout=20.0):
        """
        Go to a reference waypoint with the RIGHT arm only.
        LEFT arm stays at ZERO pose.
        """
        name = ref.get("name", "<unnamed>")
        xR_ref, RR_ref = ref_right_pose(ref)

        print(f"[TASK] go_to('{name}' , RIGHT only, tol={tol}, hold={hold})")

        # Right arm → waypoint, Left arm → zero
        spring_R.set_target(xR_ref,       RR_ref)
        spring_L.set_target(ZERO_LEFT_POS, ZERO_LEFT_ROT)

        start_time    = time.time()
        reached       = False
        reached_since = None

        while not stop_event.is_set():
            t = time.time() - start_time

            # Measure actual EE
            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()

            err_R = np.linalg.norm(xR_ref        - xR_meas)
            err_L = np.linalg.norm(ZERO_LEFT_POS - xL_meas)
            err   = max(err_R, err_L)

            if err < tol:
                if not reached:
                    reached = True
                    reached_since = time.time()
                    print(f"[TASK] '{name}' reached (err={err:.4f} m). Holding...")
                else:
                    if (time.time() - reached_since) > hold:
                        print(f"[TASK] Finished holding '{name}'.")
                        return True
            else:
                reached = False
                reached_since = None

            if t > timeout:
                print(f"[WARN] go_to('{name}') timeout (err={err:.4f} m).")
                return False

            time.sleep(CTRL_DT)

    # --- Task sequence -------------------------------------------------------
    # 1) Print initial EE pose
    xR_start, _ = robot.get_R_ee_pose()
    xL_start, _ = robot.get_L_ee_pose()

    print("[TASK] Start pose:")
    print("  Right:", xR_start)
    print("  Left :", xL_start)

    # 2) Go to zero pose
    go_to_zero(tol=0.02, hold=0.3)

    if GRIPPER_USED : 
        # gripper.set_left_gripper(LEFT_GRIPPER_OPEN)
        gripper.set_right_gripper(RIGHT_GRIPPER_OPEN)
        time.sleep(1.0)

    # 2) Ensure we have at least one waypoint
    # if not waypoints:
    #     print("[TASK] No waypoints provided; nothing to do.")
    #     stop_event.set()
    #     return

    # # 3) Follow all reference points
    # print(f"[TASK] Following {len(waypoints)} reference points from file:")
    # for idx, ref in enumerate(waypoints):
    #     if stop_event.is_set():
    #         break

    #     name = ref.get("name", f"wp_{idx:03d}")

    #     # Example: slightly larger tol/hold for "contact" points
    #     if "contact" in name.lower():
    #         tol  = 0.03
    #         hold = 1.0
    #     else:
    #         tol  = 0.03
    #         hold = 0.20
        # if "fifth" in name.lower():
        #     if GRIPPER_USED : 
        #         gripper.set_left_gripper(LEFT_GRIPPER_OPEN)
        #         gripper.set_right_gripper(RIGHT_GRIPPER_CLOSED)
        #         time.sleep(1.0)
                # gripper.set_right_gripper(RIGHT_GRIPPER_OPEN)
                # time.sleep(1.0)
                # gripper.set_right_gripper(RIGHT_GRIPPER_CLOSED)
                # time.sleep(1.0)
                # gripper.set_right_gripper(RIGHT_GRIPPER_OPEN)
                # time.sleep(1.0)
                # gripper.set_right_gripper(RIGHT_GRIPPER_CLOSED)
                # time.sleep(1.0)


        # go_to_ref_right_only(ref, tol=tol, hold=hold)

    time.sleep (0.50)
    # put both open as a safe default
    if GRIPPER_USED : 
        # gripper.set_left_gripper(LEFT_GRIPPER_OPEN)
        gripper.set_right_gripper(RIGHT_GRIPPER_CLOSED)
        time.sleep(15.0)

    # go_to_zero(tol=0.03, hold=0.2)

    print("[TASK] Task thread finished.")
    stop_event.set()

# --- 3.3 State logger thread --------------------------------------------------

def state_logger_thread(robot,gripper):
    """
    Thread that logs time-series data:
      t, EE_meas (R/L), EE_ref (R/L), joints
    """
    t0 = time.time()
    while not stop_event.is_set():
        t = time.time() - t0

        # Read measured EE poses
        xR_meas, RR_meas = robot.get_R_ee_pose()
        xL_meas, RL_meas = robot.get_L_ee_pose()

        # Read joints (motor-state dict)
        joints = robot.read_joints()

        # Read current reference from shared_state
        with state_lock:
            xR_ref = shared_state["ref_xR"].copy()
            RR_ref = shared_state["ref_RR"].copy()
            xL_ref = shared_state["ref_xL"].copy()
            RL_ref = shared_state["ref_RL"].copy()
       
        # Gripper states: actual if possible, otherwise defaults
        if GRIPPER_USED and gripper is not None:
            try:
                left_grip, right_grip = gripper.get_positions()
                # make sure they are plain floats
                left_grip  = float(left_grip)
                right_grip = float(right_grip)
            except Exception as e:
                print(f"[WARN] gripper.get_positions() failed: {e}")
                left_grip  = float(DEFAULT_LEFT_GRIPPER)
                right_grip = float(DEFAULT_RIGHT_GRIPPER)
        else:
            left_grip  = float(DEFAULT_LEFT_GRIPPER)
            right_grip = float(DEFAULT_RIGHT_GRIPPER)

        sample = {
            "t": float(t),
            "right": {
                "x_meas": np_round_list(xR_meas),
                "R_meas": np_round_list(RR_meas),
                "x_ref":  np_round_list(xR_ref),
                "R_ref":  np_round_list(RR_ref),
                "gripper": right_grip,
            },
            "left": {
                "x_meas": np_round_list(xL_meas),
                "R_meas": np_round_list(RL_meas),
                "x_ref":  np_round_list(xL_ref),
                "R_ref":  np_round_list(RL_ref),
                "gripper": left_grip,
            },
            "joints": joints,  # already a dict of floats
        }

        log_data.append(sample)
        time.sleep(LOG_DT)

    print("[LOGGER] Logger thread exiting (stop_event set).")
# --- 3.4 Video thread ---------------------------------------------------------

def frame_writer_thread(cap, out_video_path, timestamps_csv_path, stop_event: threading.Event):
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid = cv2.VideoWriter(str(out_video_path), fourcc, VIDEO_FPS, (w, h))

    with open(timestamps_csv_path, "w", newline="") as fts:
        wcsv = csv.writer(fts)
        wcsv.writerow(["frame_idx", "time_s"])
        idx = 0
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            t = time.time()
            vid.write(frame)
            wcsv.writerow([idx, f"{t:.6f}"])
            idx += 1
    vid.release()
# -----------------------------------------------------------------------------
# 4) Main
# -----------------------------------------------------------------------------

def main():
     # --- 0. Prepare run directory + paths ---
    
    base_dir   = os.path.dirname(__file__)
    RESULTS_DIR = os.path.join(base_dir, "database", TASK_NAME)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_dir    = next_run_dir(RESULTS_DIR, RUN_NAME)
    meta_path  = run_dir / "meta.json"
    json_path  = run_dir / "task_space_log.json"   # or whatever name you want
    video_path = run_dir / "camera.mp4"
    ts_csv_path = run_dir / "frame_timestamps.csv"

    print("[INFO] Run directory:", run_dir)

    # --- 1. Init camera ---
    # cap, (act_w, act_h), act_fps = open_camera(CAM_INDEX, CAM_RES, CAM_FPS_REQ)
    # if not cap.isOpened():
    #     print("[ERR] Camera failed to open.")
    #     return 2

    # --- 2. DDS init ---
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS initialized (domain={DDS_DOMAIN}, iface='{DDS_IFACE}').")
    except Exception as e:
        print("[WARN] DDS init issue:", e)
    
    

    # --- 3. Robot controller: one-layer PD ---
    robot = G1RobotArmController(
        ctrl_dt=CTRL_DT,
        ctrl_2l_dt=CTRL_2L_DT,  # None -> one-layer
        mode=MODE,
        visualize=False,
        imp=False               # IMPORTANT: no internal impedance mapping
    )

    robot.start()
    time.sleep(1.0)

    # --- 4. Build spring–damper modules for R/L ---
    spring_R = SpringDamperCartesian(
        K_p=RIGHT_SPRING_K_P, D_p=RIGHT_SPRING_D_P, M_p=RIGHT_SPRING_M_P,
        K_r=RIGHT_SPRING_K_R, D_r=RIGHT_SPRING_D_R, M_r=RIGHT_SPRING_M_R
    )
    spring_L = SpringDamperCartesian(
        K_p=LEFT_SPRING_K_P,  D_p=LEFT_SPRING_D_P,  M_p=LEFT_SPRING_M_P,
        K_r=LEFT_SPRING_K_R,  D_r=LEFT_SPRING_D_R,  M_r=LEFT_SPRING_M_R
    )

    # Initialize springs: target = ZERO, offset = current pose
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()
    spring_R.set_target(ZERO_RIGHT_POS, ZERO_RIGHT_ROT)
    spring_L.set_target(ZERO_LEFT_POS,  ZERO_LEFT_ROT)
    spring_R.reset(xR_start, RR_start)
    spring_L.reset(xL_start, RL_start)

    # --- 5. Load reference waypoints from JSON file ---
    print("[INFO] Loading reference points from:", REFERENCE_JSON_PATH)
    waypoints = load_reference_points(REFERENCE_JSON_PATH)
    print(f"[INFO] Loaded {len(waypoints)} reference points.")
    # --- 6. Save static meta (before threads) ---
    meta = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dds_iface": DDS_IFACE,
        "dds_domain": DDS_DOMAIN,
        "mode": MODE,
        "controller_dt": CTRL_DT,
        "controller_2l_dt": CTRL_2L_DT,
        "camera": {
            "index": CAM_INDEX,
            # "resolution": [act_w, act_h],
            "fps_req": CAM_FPS_REQ,
            # "fps_reported": act_fps,
        },
        "reference_file": REFERENCE_JSON_PATH,
        "task_name": TASK_NAME,
        "run_name": RUN_NAME,
        "task_discription": TASK_DESCRIPTION,
        "spring_right": {
            "K_p": list(RIGHT_SPRING_K_P),
            "D_p": list(RIGHT_SPRING_D_P),
            "M_p": list(RIGHT_SPRING_M_P),
            "K_r": list(RIGHT_SPRING_K_R),
            "D_r": list(RIGHT_SPRING_D_R),
            "M_r": list(RIGHT_SPRING_M_R),
        },
        "spring_left": {
            "K_p": list(LEFT_SPRING_K_P),
            "D_p": list(LEFT_SPRING_D_P),
            "M_p": list(LEFT_SPRING_M_P),
            "K_r": list(LEFT_SPRING_K_R),
            "D_r": list(LEFT_SPRING_D_R),
            "M_r": list(LEFT_SPRING_M_R),
        },
        "zero_pose": {
            "right": {
                "x": np_round_list(ZERO_RIGHT_POS),
                "R": np_round_list(ZERO_RIGHT_ROT),
            },
            "left": {
                "x": np_round_list(ZERO_LEFT_POS),
                "R": np_round_list(ZERO_LEFT_ROT),
            },
        },
    }
    with open(meta_path, "w") as f:
        json.dump({"meta": meta}, f, indent=2)


    # --- 6.5 Init gripper hardware (optional) ---
    if GRIPPER_USED and DynamixelController is not None:
        try:
            gripper = DynamixelController(init_left= LEFT_GRIPPER_CLOSED,init_right=RIGHT_GRIPPER_CLOSED)
            # put both open as a safe default
            gripper.set_left_gripper(LEFT_GRIPPER_CLOSED)
            gripper.set_right_gripper(RIGHT_GRIPPER_CLOSED)
            time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] Gripper init failed: {e}")
            gripper = None
    else:
        gripper = None

   
    # --- 7. Start threads: control, logger, task, video ---
    ctrl_thr   = threading.Thread(target=control_thread,
                                  args=(robot, spring_R, spring_L),
                                  daemon=True)
    logger_thr = threading.Thread(target=state_logger_thread,
                                  args=(robot,gripper),
                                  daemon=True)
    task_thr   = threading.Thread(target=task_thread,
                                  args=(robot, spring_R, spring_L, 
                                        waypoints,gripper ),
                                  daemon=False)   # main will join this
    # video_thr  = threading.Thread(target=frame_writer_thread,
    #                               args=(cap, video_path, ts_csv_path, stop_event),
    #                               daemon=True)

    ctrl_thr.start()
    logger_thr.start()
    # video_thr.start()
    task_thr.start()

    try:
        task_thr.join()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt: stopping...")
        stop_event.set()
    finally:
        stop_event.set()
        logger_thr.join(timeout=2.0)
        ctrl_thr.join(timeout=2.0)
        # video_thr.join(timeout=2.0)
        # cap.release()

    # --- 8. Save log_data to JSON ---
    print("[INFO] Saving results to:", json_path)
    out = {"samples": log_data}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    robot.stop()
    print("[INFO] Done. Robot stopped.")

if __name__ == "__main__":
    main()
