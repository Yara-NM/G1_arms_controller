#!/usr/bin/env python3
# note: edit the discription to include ALL impedance parameters for both rotation and position in the meta data.also 
# note: to save desired and measured poses dor each dt and joints. and gripper states. 
 
"""
collect_data.py

Data collection runner:
- Initializes G1 controller, zeros arms.
- Applies per-joint Kp/Kd maps + a speed preset/scale (new pkg feature).
- Initializes camera + results directory.
- Captures:
    * MP4 video with timestamps in a CSV
    * Joint states CSV (time, joints, gripper states)
- Executes a reference-points task thread:
    move to <object> -> close -> move to <plate> -> open

Edit the CONFIG section as needed.
"""
# note: add the impedance import. 
import os, sys, json, time, threading, queue, csv
from pathlib import Path
import cv2
import numpy as np
from arms_controller import G1RobotArmController  # <-- change if needed
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# ==================== CONFIG ====================

# Note: I want to keep the pathes the same. under the parent, where this file exist, there is results and under it we will create a task folder 
# note: to save all the atempts for the same task. and keep task config  discreprion. 
# note: add impedance dt to the config
TASK_FOLDER = "pick_grapes"
RESULTS_DIR= Path(__file__).resolve().parent / "results" / "runs" / TASK_FOLDER
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME         = time.strftime("run_%Y%m%d_%H%M%S")

REF_POINTS_JSON_NAME = "pick_grapes_25.json"

REF_POINTS_JSON  = Path(__file__).resolve().parent / "results" / "reference_points" / REF_POINTS_JSON_NAME
TASK_DESCRIPTION = "Grapes-to-plate trial (move→close→move→open)"

# note: do not change good. impedance will be true and ctrl_2l_dt will be None. 
# DDS / controller
DDS_DOMAIN = 0                         # 0/1  for real/sim 
DDS_IFACE  = "enp2s0"                  # "enp2s0"/"lo" for real/sim
CTRL_DT    = 0.02
CTRL_2L_DT = 0.05
MODE       = "l"

# note: Camera config. I think is good we do not need more than this. 
CAM_INDEX   = 6
CAM_RES     = (1280, 720)
CAM_FPS_REQ = 30
VIDEO_FPS   = 30     # encode target; will run as fast as camera allows

# Logging, note: we do not need the torque
STATE_DT = 0.02      # joint/state logging rate (s)
LOG_TORQUE = False   # set False to disable torque logging in states.csv

# Note: this for older pipeline now we are changing impedance parameter in cartesian space K,D,M for pose and rotation and both arms. 
# NEW FEATURES: Kp/Kd maps & speed
KP_MAP = {  # fill all arm joints; examples:
    "LeftShoulderPitch": 40.0, "LeftShoulderRoll": 40.0, "LeftShoulderYaw": 40.0,
    "LeftElbow": 35.0, "LeftWristRoll": 20.0, "LeftWristPitch": 20.0, "LeftWristYaw": 20.0,
    "RightShoulderPitch": 40.0, "RightShoulderRoll": 40.0, "RightShoulderYaw": 40.0,
    "RightElbow": 35.0, "RightWristRoll": 20.0, "RightWristPitch": 20.0, "RightWristYaw": 20.0
}
KD_MAP = {
    "LeftShoulderPitch": 1.7, "LeftShoulderRoll": 1.7, "LeftShoulderYaw": 1.7,
    "LeftElbow":1.5, "LeftWristRoll": 1.2, "LeftWristPitch": 1.2, "LeftWristYaw": 1.2,
    "RightShoulderPitch": 1.7, "RightShoulderRoll": 1.7, "RightShoulderYaw": 1.7,
    "RightElbow": 1.5, "RightWristRoll": 1.2, "RightWristPitch": 1.2, "RightWristYaw": 1.2
}
# Speed options, Note:  we do not have anything like this in this pipeline maybe we can add vmax and w max for the impedance controller.  
SPEED_PRESET = "fragile"   # "normal", "human_near", "fragile"
SPEED_SCALE  = 0.9      # 1.0 ,  0.7  , 0.9 


# Grippers note: keep the same

GRIPPER_USED   = True    # set False to disable any gripper init/commands
GRIPPER_OPEN   = 0   # 
GRIPPER_CLOSED = 170

DEFAULT_LEFT_GRIPPER  = 0  # if GRIPPER_USED == False, use these
DEFAULT_RIGHT_GRIPPER = 0

# Lazy import so we don’t touch serial if disabled note: keep the same because the connection to the gripper cause issues no need  to import if disabled 
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
# Note: we only gonna save if it's open or closed: binary
# Note: I think previosly we were reading the gripper state from the file too. now we don not. we gonna add open or close directly in the coresponding thread whitout readin the state from the file. 
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

# note: this one is not needed because the impedance already doing this (correct me if I am mistaken)
def linspace_dict(a: dict, b: dict, steps: int):
    """Interpolate joint dicts (same keys) in 'steps' steps (excluding a, including b)."""
    keys = list(b.keys())
    arr_a = np.array([float(a.get(k, 0.0)) for k in keys], dtype=float)
    arr_b = np.array([float(b[k]) for k in keys], dtype=float)
    for i in range(1, steps+1):
        alpha = i / steps
        arr = (1 - alpha) * arr_a + alpha * arr_b
        yield {k: float(v) for k, v in zip(keys, arr)}

def load_reference_points(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Reference points file not found: {json_path}")
    data = json.load(open(json_path, "r"))
    pts = data.get("points", [])
    return pts
# note: we might change the logic of the task theard. Also those function gonna change becuase we have different values saved. 
def pick_point(points, name_contains):
    """Return first point whose 'name' contains substring (case-insensitive)."""
    needle = name_contains.lower()
    for p in points:
        if needle in str(p.get("name","")).lower():
            return p
    return None

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
# note: keep this thread ) it's for recording the video. 
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

# note: this thread going to change for each task and depending on the file we read. propably the same logic. points do something with them to change refrence for impedance. 
# note: we might need to add dt for impedance sake. 
def task_runner_thread(g1, ref_json_path, stop_event: threading.Event, gripper: "GripperProxy|None" = None):
    """
    For *each* grape point (arm 'l' or 'r'):
      - move that arm to the grape TF (other arm frozen)
      - close gripper
      - move same arm to the matching plate TF (same arm)
      - open gripper
      - reset to zero
    No interpolation. Uses move_arms(left_tf, right_tf).
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
        time.sleep(g1.estimate_total_motion_time() + 0.3)




    # ---- load & index points ----
    pts = _load_points(ref_json_path)
    if not pts:
        print("[TASK] No reference points; skipping.")
        return

    # All grapes (any number), and pick plate per arm (first match)
    grapes = []
    plate_L = None
    plate_R = None
    for p in pts:
        nm  = str(p.get("name","")).lower()
        arm = str(p.get("arm","")).lower()
        if "grape" in nm:
            if arm.startswith("l"):
                grapes.append( ("l", p) )
            elif arm.startswith("r"):
                grapes.append( ("r", p) )
        elif "plate" in nm:
            if arm.startswith("l") and plate_L is None:
                plate_L = p
            if arm.startswith("r") and plate_R is None:
                plate_R = p

    if not grapes:
        print("[TASK] No 'grape' points found."); return
    if plate_L is None and any(a=="l" for a,_ in grapes):
        print("[TASK] No 'plate' point for left arm.")
    if plate_R is None and any(a=="r" for a,_ in grapes):
        print("[TASK] No 'plate' point for right arm.")

    # ---- run: for each grape ----
    for arm_char, p_grape in grapes:
        if stop_event.is_set(): break
        print(f"[TASK] Pick '{p_grape.get('name','grape')}' with arm={arm_char}")

        # Move to grape
        T_grape = _tf_for_arm(p_grape, arm_char)
        _move_to_tf_using_move_arms(T_grape, arm_char)
        
        # Close
        if gripper is not None:
            gripper.close(arm_char)
        time.sleep(2.0)

        # Move to plate (same arm)
        p_plate = plate_L if arm_char == "l" else plate_R
        T_plate = _tf_for_arm(p_plate, arm_char)

        if T_plate is None:
              print(f"[TASK] Invalid plate TF for arm={arm_char} (not 4x4).")
        else:
            print(f"[TASK] Place '{p_grape.get('name','grape')}' with arm={arm_char}")
            _move_to_tf_using_move_arms(T_plate, arm_char)

            # Open
            if gripper is not None:
                gripper.open(arm_char)
            time.sleep(2.0)

        # Back to zero
        g1.reset_arms()
        time.sleep(g1.estimate_total_motion_time() + 0.3)

    print("[TASK] Finished all grape pickups.")

#note: this thread is to log the satets repetedly we might not need it becuase we can save from impedance while running the impedance control loop. (we can discuss the need for it)   
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
    # Resultsupdate_kp
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


    # Save meta. not: will change considering new values for impedance
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
        "logging": {
            "state_dt": STATE_DT,
            "torque_logged": LOG_TORQUE,
            "torque_units": "N·m"  # adjust if your API returns other units
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Meta saved -> {meta_path}")

    # gripper

    # gripper hardware (optional)
    hw = DynamixelController() if GRIPPER_USED and DynamixelController else None
    gripper = GripperProxy(hw, log_mode="binary")

    # Threads
    stop_event = threading.Event()
    th_video = threading.Thread(target=frame_writer_thread, args=(cap, video_path, ts_csv_path, stop_event), daemon=True)
    th_state = threading.Thread(target=state_logger_thread, args=(g1, state_csv, gripper , stop_event), daemon=True)
    
    th_task = threading.Thread(target=task_runner_thread, args=(g1, REF_POINTS_JSON, stop_event, gripper),daemon=True)

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
