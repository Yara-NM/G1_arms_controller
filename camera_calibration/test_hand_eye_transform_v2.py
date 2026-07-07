"""
test_hand_eye_transform.py
--------------------------
VERIFY (not calibrate) the URDF-derived camera -> pelvis/base transform on the G1.

What it does:
  1. Boots the ONE-LAYER spring-damper stack (same as test_interactive_profiles.py).
  2. Homes both arms to the zero pose.
  3. Runs a camera thread that detects one ArUco marker, estimates its pose by
     PnP (marker size + intrinsics), and -- if a RealSense with depth is used --
     cross-checks the PnP distance against measured depth.
  4. On each ENTER keypress: snapshots the marker's base-frame position, commands
     the chosen arm to hover above it (via the spring-damper target, so the motion
     inherits your impedance PROFILE), waits to settle, and logs everything.
  5. Saves all poses to camera_calibration/results/<timestamp>.json.

Frames:
  - PnP tvec and depth-deprojected point are in the OpenCV OPTICAL frame
    (X-right, Y-down, Z-forward).
  - R_CAM_TO_BASE / T_CAM_TO_BASE (derived below from the URDF, waist locked)
    map optical -> pelvis/base, which is the same frame the IK/FK EE poses use.

IMPORTANT scope note:
  In SIMULATION the arm moves in MuJoCo while the marker is real, so the logged
  "reach error" reflects IK + impedance tracking only -- it does NOT prove the
  transform is metrically correct against the real marker. Use sim to confirm the
  base-frame position is PLAUSIBLE and reachable and the motion looks right.
  The true transform accuracy check is done on the REAL robot (flip SIM=False):
  physically see whether the hand lands on the marker.

Requires:
  - spring_damper_cartesian.py with set_gains()/set_ref_speed()  (patched version)
  - task_space_impedance/profiles.py
  - pyrealsense2  (only if CAMERA_BACKEND="realsense")
"""

import os
import json
import time
import math
import threading
from datetime import datetime

import numpy as np
# Force OpenCV's Qt GUI onto X11/XWayland instead of the wayland plugin.
# The wayland path fails to find its plugin and BLOCKS during imshow init,
# which (under the GIL) starves the 50 Hz control loop. Must be set before cv2.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian
from task_space_impedance.profiles import apply_profile, DEFAULT_PROFILE, describe

# ============================================================================
# CONFIG  (everything you'd change lives here -- no command-line args)
# ============================================================================

# ---- Sim vs real robot (one-line flip) ----
SIM         = True                 # True = MuJoCo/loopback, False = real robot
DDS_DOMAIN  = 1 if SIM else 0      # 1 = sim, 0 = real
DDS_IFACE   = "lo" if SIM else "enp2s0"
CTRL_DT     = 0.02                 # 50 Hz joint + spring-damper loop

# ---- Impedance profile for the motion ----
PROFILE     = "medium"             # "soft" / "medium" / "stiff" from profiles.py
SPEED_SCALE = 1.0

# ---- Which arm reaches the marker ----
MOVE_ARM        = "R"              # "R" or "L"
APPROACH_OFFSET = np.array([0.0, 0.0, 0.10])   # hover 10 cm ABOVE marker (base frame, +Z up)
REACH_TOL       = 0.02             # [m] "arrived" threshold for logging
REACH_TIMEOUT   = 20.0              # [s] max wait for the arm to settle

# ---- Camera ----
CAMERA_BACKEND  = "realsense"      # "realsense" (RGB+depth) or "opencv" (RGB only)
CAM_INDEX       = 6                # opencv backend only
CAM_W, CAM_H    = 1280, 720
CAM_FPS         = 30
USE_DEPTH       = True             # realsense backend only; adds depth cross-check
SHOW_CAMERA     = True             # OpenCV preview window (set False if it misbehaves)
POSITION_SOURCE = "pnp"            # "pnp" (recommended) or "depth" for the commanded 3D point

# ---- Marker ----
ARUCO_DICT_NAME = "DICT_5X5_100"   # must match make_aruco_marker.py
MARKER_ID       = 7
MARKER_LENGTH_M = 0.050            # <-- set to the MEASURED printed black-side size [m]

# ---- Intrinsics + output ----
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
INTRINSICS_JSON = os.path.join(_SCRIPT_DIR, "g1_new_camera_intrinsics_1280x720.json")
SAVE_DIR        = os.path.join(_SCRIPT_DIR, "results")

# ---- URDF-derived transform (waist LOCKED at zero) --------------------------
# d435_link relative to torso_link  (URDF joint 'd435_joint')
CAM_XYZ_IN_TORSO = np.array([0.0576235, 0.01753, 0.42987])
CAM_RPY_IN_TORSO = np.array([0.0, 0.8307767239493009, 0.0])
# torso_link relative to pelvis (waist chain collapsed at 0 -> only waist_roll origin)
TORSO_XYZ_IN_PELVIS = np.array([-0.0039635, 0.0, 0.044])
TORSO_RPY_IN_PELVIS = np.array([0.0, 0.0, 0.0])
# OpenCV optical (X-right,Y-down,Z-fwd) -> URDF link body axes
R_OPTICAL_TO_BODY = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]], dtype=float)
# ============================================================================


def build_cam_to_base():
    """Compose optical-frame -> pelvis/base transform from the URDF config above."""
    R_body_to_torso = R.from_euler('xyz', CAM_RPY_IN_TORSO).as_matrix()
    R_opt_to_torso = R_body_to_torso @ R_OPTICAL_TO_BODY
    R_torso_to_pelvis = R.from_euler('xyz', TORSO_RPY_IN_PELVIS).as_matrix()

    R_cam_to_base = R_torso_to_pelvis @ R_opt_to_torso
    t_cam_to_base = R_torso_to_pelvis @ CAM_XYZ_IN_TORSO + TORSO_XYZ_IN_PELVIS
    return R_cam_to_base, t_cam_to_base


R_CAM_TO_BASE, T_CAM_TO_BASE = build_cam_to_base()


def get_zero_poses():
    """Zero (home) EE poses for right and left arms -- same as the impedance test."""
    xR_zero = np.array([0.32686 - 0.15, -0.15184 - 0.10, 0.06149])
    RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                        [-0.01149,  0.99992, -0.00482],
                        [-0.11184,  0.00350,  0.99372]])
    xL_zero = np.array([0.32683 - 0.15,  0.15156 + 0.10, 0.06145])
    RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                        [ 0.01129,  0.99992,  0.00597],
                        [-0.11195, -0.00467,  0.99370]])
    return xR_zero, RR_zero, xL_zero, RL_zero


def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        c = json.load(f)
    K = c.get("camera_matrix") or c.get("K") or c.get("mtx")
    dist = (c.get("distortion_coefficients") or c.get("distCoeffs")
            or c.get("distortion"))
    if K is None or dist is None:
        raise KeyError(f"Intrinsics JSON missing keys. Has: {list(c.keys())}")
    return np.array(K, np.float32), np.array(dist, np.float32).reshape(-1)


# ----------------------------------------------------------------------------
# Camera backends -- common interface: read() -> BGR frame ; depth_at(u,v) -> m|None
# ----------------------------------------------------------------------------
class OpenCVCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        time.sleep(1.0)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV camera {CAM_INDEX} did not open.")
        self.has_depth = False

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def depth_at(self, u, v, win=2):
        return None

    def close(self):
        self.cap.release()


class RealSenseCamera:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        self.has_depth = USE_DEPTH
        if USE_DEPTH:
            cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if USE_DEPTH else None
        self._depth_frame = None
        # warm up
        for _ in range(10):
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.pipe.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
            self._depth_frame = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def depth_at(self, u, v, win=2):
        if self._depth_frame is None:
            return None
        vals = []
        for du in range(-win, win + 1):
            for dv in range(-win, win + 1):
                uu, vv = int(u + du), int(v + dv)
                if 0 <= uu < CAM_W and 0 <= vv < CAM_H:
                    d = self._depth_frame.get_distance(uu, vv)  # meters
                    if d > 0:
                        vals.append(d)
        return float(np.median(vals)) if vals else None

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def make_camera():
    if CAMERA_BACKEND == "realsense":
        return RealSenseCamera()
    return OpenCVCamera()


# ----------------------------------------------------------------------------
# PnP + depth deprojection
# ----------------------------------------------------------------------------
def pnp_marker(corners_4x2, K, dist):
    """Return tvec (3,) in optical frame, plus the 4-corner set used."""
    s = float(MARKER_LENGTH_M)
    objp = np.array([[-s/2,  s/2, 0], [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0], [-s/2, -s/2, 0]], np.float32)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(objp, corners_4x2.astype(np.float32), K, dist, flags=flag)
    if not ok:
        return None, None
    return tvec.reshape(3), rvec.reshape(3)


def deproject_depth(u, v, z, K, dist):
    """Pixel (u,v) + metric depth z -> 3D point in optical frame, using intrinsics."""
    if z is None:
        return None
    pt = cv2.undistortPoints(np.array([[[u, v]]], np.float32), K, dist)  # normalized
    x_n, y_n = pt[0, 0]
    return np.array([x_n * z, y_n * z, z], float)


def to_base(pos_optical):
    return R_CAM_TO_BASE @ np.asarray(pos_optical, float) + T_CAM_TO_BASE


# ----------------------------------------------------------------------------
# Shared marker state (written by camera thread, read by main thread)
# ----------------------------------------------------------------------------
MARKER = {"seen": False, "pnp_cam": None, "depth_cam": None,
          "depth_z": None, "pnp_z": None}
MARKER_LOCK = threading.Lock()

# Separate flags: the camera window must NEVER be able to stop the control loop.
CTRL_RUN  = {"on": True}          # controls the 50 Hz arm loop
CAM_RUN   = {"on": True}          # controls the camera thread only
CAM_READY = threading.Event()     # set once camera has done its heavy init + first frame


def camera_thread(K, dist):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    cam = make_camera()
    print(f"[CAM] backend={CAMERA_BACKEND} depth={getattr(cam,'has_depth',False)}")

    while CAM_RUN["on"]:
        frame = cam.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        found = False
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if int(mid) != MARKER_ID:
                    continue
                c = np.asarray(corners[i]).reshape(4, 2)
                tvec, _ = pnp_marker(c, K, dist)
                if tvec is None:
                    continue
                u, v = c.mean(axis=0)
                z_depth = cam.depth_at(int(u), int(v)) if getattr(cam, "has_depth", False) else None
                depth_pt = deproject_depth(u, v, z_depth, K, dist)
                with MARKER_LOCK:
                    MARKER.update(seen=True, pnp_cam=tvec, pnp_z=float(tvec[2]),
                                  depth_cam=depth_pt, depth_z=z_depth)
                found = True

                if SHOW_CAMERA:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    lbl = f"ID{MARKER_ID} pnpZ={tvec[2]:.3f}m"
                    if z_depth is not None:
                        lbl += f" depthZ={z_depth:.3f}m dz={tvec[2]-z_depth:+.3f}"
                    cv2.putText(frame, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                break
        if not found:
            with MARKER_LOCK:
                MARKER["seen"] = False

        if SHOW_CAMERA:
            if not found:
                cv2.putText(frame, f"marker {MARKER_ID} not seen", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("transform check (q here closes preview only)", frame)
            # First imshow triggers the heavy Qt GUI init. Because the camera
            # thread is started only AFTER the arm is standstill at zero, any
            # stall here happens on a static target -> no lurch.
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                CAM_RUN["on"] = False   # closes preview ONLY; controller keeps running

        # Signal readiness after the first fully-processed frame (incl. GUI init).
        CAM_READY.set()

    cam.close()
    if SHOW_CAMERA:
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("[XFORM] R_cam_to_base=\n", np.round(R_CAM_TO_BASE, 5))
    print("[XFORM] t_cam_to_base=", np.round(T_CAM_TO_BASE, 5))

    K, dist = load_intrinsics(INTRINSICS_JSON)

    # --- DDS + one-layer controller (same stack as the impedance test) ---
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS init (domain={DDS_DOMAIN}, iface={DDS_IFACE}).")
    except Exception as e:
        print("[WARN] DDS init:", e)

    robot = G1RobotArmController(ctrl_dt=CTRL_DT, ctrl_2l_dt=None,
                                 mode='l', visualize=False, imp=False)
    robot.start()
    time.sleep(5.0)

    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()
    xR_ref, RR_ref, xL_ref, RL_ref = get_zero_poses()

    spring_R = SpringDamperCartesian()
    spring_L = SpringDamperCartesian()
    apply_profile(spring_R, PROFILE, speed_scale=SPEED_SCALE)
    apply_profile(spring_L, PROFILE, speed_scale=SPEED_SCALE)
    print(f"[INFO] Profile: {describe(PROFILE)}")

    spring_R.set_target(xR_ref, RR_ref); spring_R.reset(xR_start, RR_start)
    spring_L.set_target(xL_ref, RL_ref); spring_L.reset(xL_start, RL_start)

    ref = {"R": xR_ref.copy(), "L": xL_ref.copy(),
           "RR": RR_ref.copy(), "RL": RL_ref.copy()}
    ref_lock = threading.Lock()

    def control_loop():
        while CTRL_RUN["on"]:
            with ref_lock:
                spring_R.set_target(ref["R"], ref["RR"])
                spring_L.set_target(ref["L"], ref["RL"])
            xR_d, RR_d, _ = spring_R.step(CTRL_DT)
            xL_d, RL_d, _ = spring_L.step(CTRL_DT)
            robot.move_arms_with_Rt(left_R=RL_d, left_t=xL_d,
                                    right_R=RR_d, right_t=xR_d)
            time.sleep(CTRL_DT)

    # === Start ONLY the control loop first. The camera thread is deliberately
    # started later, after the arm is standstill at zero, so its heavy GUI/
    # RealSense init cannot starve the control loop mid-homing. ===
    ctrl_t = threading.Thread(target=control_loop, daemon=True)
    ctrl_t.start()

    # --- home to zero (control loop does the motion) ---
    print("[INFO] Homing to zero pose...")
    t0 = time.time()
    while time.time() - t0 < REACH_TIMEOUT:
        xR_m, _ = robot.get_R_ee_pose()
        xL_m, _ = robot.get_L_ee_pose()
        if (np.linalg.norm(ref["R"] - xR_m) < REACH_TOL and
                np.linalg.norm(ref["L"] - xL_m) < REACH_TOL):
            break
        time.sleep(0.1)

    # --- confirm STANDSTILL: require the arm to stay put for a sustained window,
    #     so the camera's heavy init lands on a truly static target ---
    print("[INFO] Reached zero; confirming standstill...")
    STILL_HOLD = 1.5   # s of continuous low motion required
    STILL_TOL  = 0.003 # m frame-to-frame movement threshold
    prev, still_t = robot.get_R_ee_pose()[0], time.time()
    while time.time() - still_t < STILL_HOLD:
        time.sleep(0.05)
        cur = robot.get_R_ee_pose()[0]
        if np.linalg.norm(cur - prev) > STILL_TOL:
            still_t = time.time()   # moved -> reset the standstill timer
        prev = cur
    print("[INFO] Arm standstill at zero.")

    # --- user gate: nothing heavy starts until you're ready ---
    input("[START] Press ENTER to start the camera (arm will hold zero)... ")

    # --- NOW start the camera; wait until it has finished init + first frame ---
    cam_t = threading.Thread(target=camera_thread, args=(K, dist), daemon=True)
    cam_t.start()
    print("[INFO] Waiting for camera to be ready...")
    if not CAM_READY.wait(timeout=15.0):
        print("[WARN] Camera not ready after 15 s; continuing anyway.")
    else:
        print("[INFO] Camera ready.")

    get_ee = robot.get_R_ee_pose if MOVE_ARM == "R" else robot.get_L_ee_pose
    R_home = RR_ref if MOVE_ARM == "R" else RL_ref

    records = []
    print("\n=== Ready ===")
    print("  ENTER = capture marker + move arm there | 'h' = re-home | 'q' = quit\n")

    try:
        while True:
            cmd = input(f"[pose {len(records)+1}] ENTER/h/q: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "h":
                with ref_lock:
                    ref["R"], ref["RR"] = xR_ref.copy(), RR_ref.copy()
                    ref["L"], ref["RL"] = xL_ref.copy(), RL_ref.copy()
                print("[INFO] Re-homing.")
                continue

            with MARKER_LOCK:
                if not MARKER["seen"]:
                    print("[WARN] Marker not currently detected -- reposition and retry.")
                    continue
                pnp_cam = None if MARKER["pnp_cam"] is None else MARKER["pnp_cam"].copy()
                depth_cam = None if MARKER["depth_cam"] is None else MARKER["depth_cam"].copy()
                pnp_z, depth_z = MARKER["pnp_z"], MARKER["depth_z"]

            # choose source for the commanded 3D point
            src_cam = depth_cam if (POSITION_SOURCE == "depth" and depth_cam is not None) else pnp_cam
            marker_base = to_base(src_cam)
            target = marker_base + APPROACH_OFFSET

            print(f"[CAP] marker_base = {np.round(marker_base,3)}  "
                  f"(pnpZ={pnp_z:.3f}"
                  + (f", depthZ={depth_z:.3f}, dz={pnp_z-depth_z:+.3f} m" if depth_z else ", depth=NA")
                  + ")")
            print(f"[MOVE] {MOVE_ARM} -> hover target {np.round(target,3)}")

            ee_before, _ = get_ee()
            with ref_lock:
                ref[MOVE_ARM] = target.copy()
                ref["RR" if MOVE_ARM == "R" else "RL"] = R_home.copy()

            # wait to settle
            t0 = time.time()
            reached = False
            while time.time() - t0 < REACH_TIMEOUT:
                ee_now, _ = get_ee()
                if np.linalg.norm(target - ee_now) < REACH_TOL:
                    reached = True
                    break
                time.sleep(0.05)
            ee_after, _ = get_ee()

            # --- metrics ---
            # controller_error: EE vs what it was COMMANDED (target = marker + offset).
            #   -> "did the arm reach the command" (pure control/IK quality)
            ctrl_vec = ee_after - target
            ctrl_err = float(np.linalg.norm(ctrl_vec))
            # marker_to_ee: EE vs the ACTUAL marker (offset NOT removed).
            #   -> "how close is the fingertip to the object" (with a hover offset,
            #      this should read approximately the offset magnitude)
            m2ee_vec = ee_after - marker_base
            m2ee_err = float(np.linalg.norm(m2ee_vec))
            print(f"[DONE] reached={reached}")
            print(f"       controller_err = {ctrl_err*100:5.2f} cm  vec(cm)={np.round(ctrl_vec*100,2)} [x_fwd,y_left,z_up]")
            print(f"       marker_to_ee   = {m2ee_err*100:5.2f} cm  vec(cm)={np.round(m2ee_vec*100,2)}\n")

            records.append({
                "index": len(records) + 1,
                "timestamp": datetime.now().isoformat(),
                "sim": SIM, "profile": PROFILE, "arm": MOVE_ARM,
                "marker_id": MARKER_ID,
                "position_source": POSITION_SOURCE,
                "pnp_pos_cam": None if pnp_cam is None else pnp_cam.tolist(),
                "depth_pos_cam": None if depth_cam is None else depth_cam.tolist(),
                "pnp_z": pnp_z, "depth_z": depth_z,
                "pnp_minus_depth_z": (None if depth_z is None else pnp_z - depth_z),
                "marker_pos_base": marker_base.tolist(),
                "commanded_target_base": target.tolist(),
                "ee_before_base": ee_before.tolist(),
                "ee_after_base": ee_after.tolist(),
                "reached": reached,
                "controller_error_vec": ctrl_vec.tolist(),
                "controller_error_m": ctrl_err,
                "marker_to_ee_vec": m2ee_vec.tolist(),
                "marker_to_ee_m": m2ee_err,
            })
    except KeyboardInterrupt:
        pass

    # --- shutdown: freeze the reference at the CURRENT measured pose so the last
    #     command the arm receives is "stay here", then stop -> no lurch ---
    xR_now, RR_now = robot.get_R_ee_pose()
    xL_now, RL_now = robot.get_L_ee_pose()
    with ref_lock:
        ref["R"], ref["RR"] = xR_now.copy(), RR_now.copy()
        ref["L"], ref["RL"] = xL_now.copy(), RL_now.copy()
    time.sleep(0.2)   # let the loop apply the hold once

    CAM_RUN["on"] = False
    CTRL_RUN["on"] = False
    ctrl_t.join(timeout=2.0)
    if 'cam_t' in locals():
        cam_t.join(timeout=2.0)
    robot.stop()

    # --- console summary: systematic bias (direction), not just magnitude ---
    if records:
        cvecs = np.array([r["controller_error_vec"] for r in records])
        mvecs = np.array([r["marker_to_ee_vec"] for r in records])
        n_ok = sum(r["reached"] for r in records)
        print("\n===== SUMMARY =====")
        print(f" poses: {len(records)}   reached: {n_ok}/{len(records)}   offset(cm)={np.round(APPROACH_OFFSET*100,1)}")
        print(f" controller_err  mean|.|={np.linalg.norm(cvecs,axis=1).mean()*100:5.2f} cm   "
              f"mean vec(cm)={np.round(cvecs.mean(axis=0)*100,2)} [x_fwd,y_left,z_up]")
        print(f" marker_to_ee    mean|.|={np.linalg.norm(mvecs,axis=1).mean()*100:5.2f} cm   "
              f"mean vec(cm)={np.round(mvecs.mean(axis=0)*100,2)}")
        print(" (consistent-sign vec = systematic bias; scattered signs = random tracking)")
        print("===================\n")

    out = {
        "meta": {
            "saved": datetime.now().isoformat(),
            "sim": SIM, "profile": PROFILE, "arm": MOVE_ARM,
            "camera_backend": CAMERA_BACKEND, "use_depth": USE_DEPTH,
            "intrinsics_json": INTRINSICS_JSON,
            "marker": {"dict": ARUCO_DICT_NAME, "id": MARKER_ID,
                       "length_m": MARKER_LENGTH_M},
            "R_cam_to_base": R_CAM_TO_BASE.tolist(),
            "t_cam_to_base": T_CAM_TO_BASE.tolist(),
            "approach_offset": APPROACH_OFFSET.tolist(),
        },
        "poses": records,
    }
    out_path = os.path.join(SAVE_DIR, f"transform_check_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Saved {len(records)} pose(s) to: {out_path}")


if __name__ == "__main__":
    main()