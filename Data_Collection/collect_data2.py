#!/usr/bin/env python3
"""
collect_data2.py

Sequential reference point execution runner:
- Initializes G1 controller, zeros arms.
- Applies per-joint Kp/Kd maps + a speed preset/scale (new pkg feature).
- Initializes camera + results directory.
- Captures:
    * MP4 video with timestamps in a CSV
    * Joint states CSV (time, joints, gripper states)
- Executes ALL reference points sequentially:
    For each point in the reference points file, move to that pose and hold for specified duration

Edit the CONFIG section as needed.
"""

import os, sys, json, time, threading, queue, csv
from pathlib import Path
import cv2
import numpy as np
from arms_controller import G1RobotArmController  # <-- change if needed
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# ==================== CONFIG ====================

TASK_FOLDER = "bottle_handover"
RESULTS_DIR= Path(__file__).resolve().parent / "results" / "runs" / TASK_FOLDER
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME         = time.strftime("run_%Y%m%d_%H%M%S")

REF_POINTS_JSON_NAME = "bottle_handover.json"

REF_POINTS_JSON  = Path(__file__).resolve().parent / "results" / "reference_points" / REF_POINTS_JSON_NAME
TASK_DESCRIPTION = "Sequential execution of all reference points"


# DDS / controller
DDS_DOMAIN = 0                         # 0/1  for real/sim 
DDS_IFACE  = "enp2s0"                  # "enp2s0"/"lo" for real/sim
CTRL_DT    = 0.02
CTRL_2L_DT = 0.05
MODE       = "l"

# Camera
CAM_INDEX   = 6
CAM_RES     = (1280, 720)
CAM_FPS_REQ = 30
VIDEO_FPS   = 30     # encode target; will run as fast as camera allows

# Loggingfragile
STATE_DT = 0.02      # joint/state logging rate (s)
LOG_TORQUE = True   # set False to disable torque logging in states.csv

# NEW FEATURES: Kp/Kd maps & speed
KP_MAP = {  # fill all arm joints; examples:
    "LeftShoulderPitch": 50.0, "LeftShoulderRoll": 60.0, "LeftShoulderYaw": 50.0,
    "LeftElbow": 40.0, "LeftWristRoll": 30.0, "LeftWristPitch": 30.0, "LeftWristYaw": 32.0,
    "RightShoulderPitch": 50.0, "RightShoulderRoll": 60.0, "RightShoulderYaw": 50.0,
    "RightElbow": 40.0, "RightWristRoll": 30.0, "RightWristPitch": 30.0, "RightWristYaw": 32.0
}
KD_MAP = {
    "LeftShoulderPitch": 1.2, "LeftShoulderRoll": 1.0, "LeftShoulderYaw":1.2,
    "LeftElbow":0.8, "LeftWristRoll": 0.8, "LeftWristPitch": 0.5, "LeftWristYaw": 0.7,
    "RightShoulderPitch": 1.2, "RightShoulderRoll": 1.0, "RightShoulderYaw": 1.2,
    "RightElbow": 0.8, "RightWristRoll": 0.8, "RightWristPitch": 0.5, "RightWristYaw": 0.7
}
# Speed options: 
SPEED_PRESET = "human-near"   # "normal", "human_near", "fragile",  
SPEED_SCALE  = 0.7     # 1.0 ,  0.7  , 0.9 , 

# Sequential execution settings
HOLD_DURATION = 0.02    # seconds to hold each pose
RESET_BETWEEN_POINTS = False  # reset to zero between each point
SORT_POINTS_BY_NAME = True   # sort points alphabetically by name

# Grippers 
GRIPPER_USED   = True    # set False to disable any gripper init/commands
GRIPPER_OPEN   = 0   # 
GRIPPER_CLOSED = 40

DEFAULT_LEFT_GRIPPER  = 0  # if GRIPPER_USED == False, use these
DEFAULT_RIGHT_GRIPPER = 0

# Lazy import so we don't touch serial if disabled
DynamixelController = None
if GRIPPER_USED:
    try:
        from gripper_controller import DynamixelController
    except Exception as e:
        print(f"[WARN] Gripper disabled (import failed): {e}")
        GRIPPER_USED = False
        DynamixelController = None

# =================================================
# --- Thread-safe gripper wrapper --------------------------------------------
class GripperProxy:
    """
    Thread-safe facade:
      - Stores logical states (0=open, 1=closed) under a lock
      - Optionally drives hardware (DynamixelController) when present
      - Provides get_log_values() for the state logger (no serial reads there)
    """
    def __init__(self, hw=None, log_mode="binary"):
        self.hw = hw
        self.mode = log_mode  # "binary" or "position" (position only if you *really* want to poll ESP)
        self._lock = threading.Lock()
        self._left = 0   # 0=open; start open by convention
        self._right = 0
        # Initialize hardware once (open both)
        if self.hw:
            try:
                self.hw.set_left_gripper(GRIPPER_OPEN)
                self.hw.set_right_gripper(GRIPPER_OPEN)
            except Exception as e:
                print(f"[WARN] Gripper HW init failed: {e}")

    def open(self, arm: str):
        arm = arm.lower()
        with self._lock:
            if arm.startswith("l"):
                self._left = 0
                if self.hw:
                    try: self.hw.set_left_gripper(GRIPPER_OPEN)
                    except Exception as e: print(f"[WARN] set_left_gripper open failed: {e}")
            else:
                self._right = 0
                if self.hw:
                    try: self.hw.set_right_gripper(GRIPPER_OPEN)
                    except Exception as e: print(f"[WARN] set_right_gripper open failed: {e}")

    def close(self, arm: str):
        arm = arm.lower()
        with self._lock:
            if arm.startswith("l"):
                self._left = 1
                if self.hw:
                    try: self.hw.set_left_gripper(GRIPPER_CLOSED)
                    except Exception as e: print(f"[WARN] set_left_gripper close failed: {e}")
            else:
                self._right = 1
                if self.hw:
                    try: self.hw.set_right_gripper(GRIPPER_CLOSED)
                    except Exception as e: print(f"[WARN] set_right_gripper close failed: {e}")

    def set_from_recorded_states(self, left_state: int, right_state: int):
        """
        Set gripper states based on recorded binary states (0=open, 1=closed).
        left_state, right_state: 0 or 1
        """
        with self._lock:
            self._left = int(left_state)
            self._right = int(right_state)
            
            if self.hw:
                try:
                    if left_state == 0:
                        self.hw.set_left_gripper(GRIPPER_OPEN)
                    else:
                        self.hw.set_left_gripper(GRIPPER_CLOSED)
                        
                    if right_state == 0:
                        self.hw.set_right_gripper(GRIPPER_OPEN)
                    else:
                        self.hw.set_right_gripper(GRIPPER_CLOSED)
                except Exception as e:
                    print(f"[WARN] Failed to set grippers from recorded states: {e}")

    def get_log_values(self):
        """
        What to record in states.csv:
        - "binary": (0/1, 0/1) from internal flags (default; no serial I/O)
        - "position": query hardware (only if you want absolute positions)
        """
        if self.mode == "position" and self.hw:
            try:
                return self.hw.get_positions()  # (left_pos, right_pos)
            except Exception as e:
                print(f"[WARN] gripper get_positions failed: {e}")
        with self._lock:
            return int(self._left), int(self._right)

# =================================================
# ---------- helpers ----------
def init_dds():
    ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
    print(f"[DDS] Using iface='{DDS_IFACE}', domain={DDS_DOMAIN}")

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

def next_run_dir(root: Path, name: str):
    d = root / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d

def _quat_to_R(q):
    q = [float(v) for v in q]
    if abs(q[0]) >= max(abs(q[1]), abs(q[2]), abs(q[3])):
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n > 1e-12:
        w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),  1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),  2*(y*z + x*w),  1-2*(x*x+y*y)],
    ], dtype=float)

def _to_hmat(pos, rot):
    p = np.asarray(pos, dtype=float).reshape(3)
    r = np.asarray(rot).reshape(-1)
    if r.size == 9:
        R = r.reshape(3,3)
    elif r.size == 4:
        R = _quat_to_R(r)
    else:
        # assume already 3x3
        R = r.reshape(3,3)
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = p
    return H

# ---------- threads ----------
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

def sequential_task_runner_thread(g1, ref_json_path, stop_event: threading.Event, gripper: "GripperProxy|None" = None):
    """
    Execute ALL reference points sequentially:
    - Load all points from the reference file
    - Sort them by name (if enabled)
    - For each point: move to pose, hold for specified duration
    - Optionally reset to zero between points
    """
    time.sleep(2) 
    
    def _load_points(path):
        p = Path(path)
        if not p.exists():
            print(f"[TASK] Reference points not found: {p}")
            return []
        try:
            data = json.load(open(p, "r"))
            return data.get("points", [])
        except Exception as e:
            print(f"[TASK] Failed to read reference points: {e}")
            return []

    def _tf_for_arm(pt, arm_char):
        """Return 4x4 numpy TF for the requested arm from a saved point."""
        if arm_char == "l":
            T = pt.get("left_ee_T_world") or pt.get("right_ee_T_world")
        else:
            T = pt.get("right_ee_T_world") or pt.get("left_ee_T_world")
        if T is None: return None
        T = np.asarray(T, dtype=float)
        if T.size != 16: return None
        return T.reshape(4,4)

    def _move_to_tf_using_move_arms(target_tf_for_moving_arm, moving_arm: str):
        """Freeze other arm at current EE pose, move only the chosen arm."""
        if target_tf_for_moving_arm is None:
            print("[TASK] Missing/invalid TF; skipping move.")
            return
        if moving_arm == "l":
            R_pos, R_rot = g1.get_R_ee_pose(); R_H = _to_hmat(R_pos, R_rot)
            L_H = target_tf_for_moving_arm
        else:
            L_pos, L_rot = g1.get_L_ee_pose(); L_H = _to_hmat(L_pos, L_rot)
            R_H = target_tf_for_moving_arm
        g1.move_arms(L_H, R_H)
        time.sleep(g1.estimate_total_motion_time())

    # ---- load & sort points ----
    pts = _load_points(ref_json_path)
    if not pts:
        print("[TASK] No reference points; skipping.")
        return

    # Sort points by name if enabled
    if SORT_POINTS_BY_NAME:
        pts = sorted(pts, key=lambda p: str(p.get("name", "")))
        print(f"[TASK] Sorted {len(pts)} points alphabetically by name")

    print(f"[TASK] Starting sequential execution of {len(pts)} reference points")
    print(f"[TASK] Hold duration: {HOLD_DURATION}s, Reset between points: {RESET_BETWEEN_POINTS}")

    # ---- run: for each point ----
    for _ in range (5):
        for i, point in enumerate(pts):
            if stop_event.is_set(): 
                break
                
            point_name = point.get("name", f"point_{i}")
            point_arm = point.get("arm", "l")
            print(f"[TASK] [{i+1}/{len(pts)}] Executing '{point_name}' with arm={point_arm}")

            # Set grippers based on recorded states
            if gripper is not None:
                left_gripper_state = point.get("left_gripper_state", 0)
                right_gripper_state = point.get("right_gripper_state", 0)
                gripper.set_from_recorded_states(left_gripper_state, right_gripper_state)
                print(f"[TASK] Set grippers from recorded states: L:{left_gripper_state} R:{right_gripper_state}")

            # Move to the point
            T_target = _tf_for_arm(point, point_arm)
            if T_target is None:
                print(f"[TASK] Invalid TF for point '{point_name}'; skipping")
                continue
                
            _move_to_tf_using_move_arms(T_target, point_arm)
            
            # Hold the pose
            print(f"[TASK] Holding pose '{point_name}' for {HOLD_DURATION}s")
            time.sleep(HOLD_DURATION)

            # Reset to zero between points (if enabled)
            if RESET_BETWEEN_POINTS and i < len(pts) - 1:  # Don't reset after last point
                print(f"[TASK] Resetting to zero before next point")
                g1.reset_arms()
                time.sleep(g1.estimate_total_motion_time() + 0.3)

    g1.reset_arms()
    time.sleep(g1.estimate_total_motion_time() + 0.3)
    print("[TASK] Finished sequential execution of all reference points.")
    
def state_logger_thread(g1: G1RobotArmController, state_csv_path, gripper: "GripperProxy|None", stop_event: threading.Event):
    # Determine stable joint order once
    probe = g1.read_joints()
    joint_order = list(probe.keys())

    # Build CSV header
    pos_cols = joint_order
    tq_cols  = [f"tau_{j}" for j in joint_order] if LOG_TORQUE else []
    header   = ["time_s"] + pos_cols + tq_cols + ["LeftGripper", "RightGripper"]

    with open(state_csv_path, "w", newline="") as fs:
        wcsv = csv.writer(fs)
        wcsv.writerow(header)

        t_next = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < t_next:
                time.sleep(t_next - now)
            t_next += STATE_DT

            # Read state
            try:
                jd = g1.read_joints()  # {joint: position}
            except Exception:
                jd = {}

            # Optional: read torque once per cycle
            if LOG_TORQUE:
                try:
                    td = g1.read_torque()  # {joint: torque_in_Nm} or similar
                except Exception:
                    td = {}
            else:
                td = {}

            # Timestamp
            t = time.time()

            # Gripper logs (no serial reads here)
            if gripper is None:
                left_grip  = int(DEFAULT_LEFT_GRIPPER)
                right_grip = int(DEFAULT_RIGHT_GRIPPER)
            else:
                left_grip, right_grip = gripper.get_log_values()

            # Compose row in fixed order
            pos_vals = [float(jd.get(j, 0.0)) for j in joint_order]
            tq_vals  = [float(td.get(j, 0.0)) for j in joint_order] if LOG_TORQUE else []
            row = [f"{t:.6f}"] + pos_vals + tq_vals + [left_grip, right_grip]
            wcsv.writerow(row)

# ---------- main ----------
def main():
    # Results
    run_dir = next_run_dir(RESULTS_DIR, RUN_NAME)
    meta_path   = run_dir / "meta.json"
    video_path  = run_dir / "camera.mp4"
    ts_csv_path = run_dir / "frame_timestamps.csv"
    state_csv   = run_dir / "states.csv"

    # DDS + controller
    init_dds()
    g1 = G1RobotArmController(ctrl_dt=CTRL_DT, 
                              ctrl_2l_dt=CTRL_2L_DT,
                              results_dir=str(run_dir), 
                              mode=MODE, 
                              visualize=False)
    time.sleep(0.3)
    g1.start()
    print (f"[INFO] starting control loop with the robot")
    time.sleep(0.1)
    g1.reset_arms()
    print (f"[INFO] reseting the arms to zero pose")
    time.sleep(g1.estimate_total_motion_time()+0.5)
    time.sleep (6)

    # Apply gains + speed profile
    try: 
        g1.apply_task_profile(preset  = SPEED_PRESET, kp = KP_MAP, kd = KD_MAP,
                            lam = SPEED_SCALE, auto_damping = None,
                            kp_slew = None, kd_slew= None)
    except Exception as e:
         print(f"[WARN] Applying the task profile is not avaliable ({e}). Skipping.")

    # Camera
    cap, (act_w, act_h), act_fps = open_camera(CAM_INDEX, CAM_RES, CAM_FPS_REQ)
    if not cap.isOpened():
        print("[ERR] Camera failed to open.")
        return 2

    # Save meta
    meta = {
        "run_name": RUN_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_description": TASK_DESCRIPTION,
        "dds_iface": DDS_IFACE,
        "dds_domain": DDS_DOMAIN,
        "mode": MODE,
        "ctrl_dt": CTRL_DT,
        "ctrl_2l_dt": CTRL_2L_DT,
        "camera": {
            "index": CAM_INDEX, "resolution": [act_w, act_h], "fps_req": CAM_FPS_REQ, "fps_reported": act_fps
        },
        "gripper_used": GRIPPER_USED,
        "default_grippers_if_unused": {"LeftGripper": DEFAULT_LEFT_GRIPPER, "RightGripper": DEFAULT_RIGHT_GRIPPER},
        "kp_map": KP_MAP,
        "kd_map": KD_MAP,
        "speed": {"preset": SPEED_PRESET, "scale": SPEED_SCALE},
        "reference_points_file": str(REF_POINTS_JSON),
        "sequential_settings": {
            "hold_duration": HOLD_DURATION,
            "reset_between_points": RESET_BETWEEN_POINTS,
            "sort_points_by_name": SORT_POINTS_BY_NAME
        },
        "logging": {
            "state_dt": STATE_DT,
            "torque_logged": LOG_TORQUE,
            "torque_units": "N·m"  # adjust if your API returns other units
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Meta saved -> {meta_path}")

    # gripper hardware (optional)
    hw = DynamixelController() if GRIPPER_USED and DynamixelController else None
    gripper = GripperProxy(hw, log_mode="binary")

    # Threads
    stop_event = threading.Event()
    th_video = threading.Thread(target=frame_writer_thread, args=(cap, video_path, ts_csv_path, stop_event), daemon=True)
    th_state = threading.Thread(target=state_logger_thread, args=(g1, state_csv, gripper , stop_event), daemon=True)
    
    th_task = threading.Thread(target=sequential_task_runner_thread, args=(g1, REF_POINTS_JSON, stop_event, gripper),daemon=True)

    th_video.start(); th_state.start(); th_task.start()
    print(f"[RUN] Logging to: {run_dir}")
    print("      Ctrl+C to stop.\n")

    try:
        while th_task.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[RUN] Interrupted by user.")
    finally:
        stop_event.set()
        th_task.join(timeout=1.0)
        th_state.join(timeout=1.0)
        th_video.join(timeout=1.0)
        cap.release()

    print(f"[DONE] Saved:\n  - Video: {video_path}\n  - Frame timestamps: {ts_csv_path}\n  - States: {state_csv}\n  - Meta: {meta_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
