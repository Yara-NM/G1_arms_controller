"""
test_grasp_orientation.py
--------------------------
Hardware test for compute_grasp_orientation() (see grasp_orientation.py),
using the SAME one-layer PD + SpringDamperCartesian stack as
test_interactive_profiles.py / test_go_to_zero.py.

Sequence:
  1. DDS init, boot G1RobotArmController (one-layer PD, imp=False).
  2. Home BOTH arms to zero pose, fixed profile (default: medium).
  3. For each arm in [R, L]:
       For each mode in [palm_down, inward]:
         - re-home that arm to zero
         - for each test offset (+/-10cm, +/-20cm on x/y/z, a couple combos):
             compute R_target via compute_grasp_orientation()
             push new reference for THAT ARM ONLY
                 (the other arm's ref is never touched here, so it just
                 sits at its last zero-pose reference the whole time)
             wait for convergence, print tracking error
             re-home that arm to zero
  Manual Enter-to-continue gate before every commanded move, so you can
  watch the robot and abort (Ctrl+C) before anything compounds.

Requires grasp_orientation.py to be importable (same directory, or add its
folder to sys.path below).
"""

import time
import os
import sys
import threading
import csv
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian
from task_space_impedance.profiles import apply_profile

# sys.path.insert(0, os.path.dirname(__file__))  # uncomment if needed
from grasp_orientation import compute_grasp_orientation, ZERO

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DDS_DOMAIN = 1          # 1 = sim/loopback, 0 = real robot
DDS_IFACE  = "lo"       # "lo" for sim, "enp2s0" for real robot
CTRL_DT    = 0.02       # 50 Hz joint + spring-damper loop
SPEED_SCALE = 1.0
TEST_PROFILE = "medium"     # fixed profile for the whole orientation test
POS_TOL_M = 0.02            # [m] "converged" tolerance
SETTLE_TIMEOUT_S = 6.0

TEST_OFFSETS = {
    'x+10cm': np.array([ 0.10,  0.0,   0.0]),
    'x-10cm': np.array([-0.10,  0.0,   0.0]),
    'y+10cm': np.array([ 0.0,   0.10,  0.0]),
    'y-10cm': np.array([ 0.0,  -0.10,  0.0]),
    'z+10cm': np.array([ 0.0,   0.0,   0.10]),
    'z-10cm': np.array([ 0.0,   0.0,  -0.10]),
    'x+20cm': np.array([ 0.20,  0.0,   0.0]),
    'z-20cm': np.array([ 0.0,   0.0,  -0.20]),
}


def pause(msg="Press Enter to continue (Ctrl+C to abort)..."):
    input(msg)


def main():
    # === 0. DDS init ===
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS initialized (domain={DDS_DOMAIN}, iface={DDS_IFACE}).")
    except Exception as e:
        print("[WARN] DDS init issue:", e)

    # === 1. One-layer PD controller only (no torque/impedance pipeline) ===
    robot = G1RobotArmController(
        ctrl_dt=CTRL_DT,
        ctrl_2l_dt=None,   # force one-layer controller
        mode='l',          # low-level
        visualize=False,
        imp=False,
    )
    robot.start()
    time.sleep(5.0)  # let encoders & DDS settle

    # === 2. Start pose ===
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()
    print("[INFO] Start pose:")
    print("  Right pos:", xR_start)
    print("  Left  pos:", xL_start)

    # === 3. Zero pose -- reuse grasp_orientation.ZERO (same numbers as
    #         get_zero_poses() elsewhere) so there's a single source of truth
    xR_zero, RR_zero = ZERO['R']['x'], ZERO['R']['R']
    xL_zero, RL_zero = ZERO['L']['x'], ZERO['L']['R']

    # === 4. Spring-damper modules, one FIXED profile for the whole test ===
    spring_R = SpringDamperCartesian()
    spring_L = SpringDamperCartesian()
    lbl = apply_profile(spring_R, TEST_PROFILE, speed_scale=SPEED_SCALE)
    apply_profile(spring_L, TEST_PROFILE, speed_scale=SPEED_SCALE)
    print(f"[INFO] Fixed profile for this test: {lbl}")

    spring_R.set_target(xR_zero, RR_zero)
    spring_L.set_target(xL_zero, RL_zero)
    spring_R.reset(xR_start, RR_start)
    spring_L.reset(xL_start, RL_start)

    # === 5. Shared state for the control-loop thread ===
    run_flag = {"running": True}
    ref_lock = threading.Lock()
    ref = {
        "R": {"x": xR_zero.copy(), "R": RR_zero.copy()},
        "L": {"x": xL_zero.copy(), "R": RL_zero.copy()},
    }

    # === 6. Logging ===
    log_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "grasp_orientation_test_log.csv")
    print(f"[INFO] Logging EE positions to: {log_path}")

    # === 7. Control loop thread (50 Hz: spring -> IK -> joint PD) ===
    def control_loop():
        t0 = time.time()
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "xR_meas_x", "xR_meas_y", "xR_meas_z",
                "xL_meas_x", "xL_meas_y", "xL_meas_z",
                "xR_des_x", "xR_des_y", "xR_des_z",
                "xL_des_x", "xL_des_y", "xL_des_z",
            ])
            while run_flag["running"]:
                dt = CTRL_DT
                t = time.time() - t0

                with ref_lock:
                    spring_R.set_target(ref["R"]["x"], ref["R"]["R"])
                    spring_L.set_target(ref["L"]["x"], ref["L"]["R"])

                xR_d, RR_d, _ = spring_R.step(dt)
                xL_d, RL_d, _ = spring_L.step(dt)

                robot.move_arms_with_Rt(
                    left_R=RL_d, left_t=xL_d,
                    right_R=RR_d, right_t=xR_d,
                )

                xR_meas, _ = robot.get_R_ee_pose()
                xL_meas, _ = robot.get_L_ee_pose()
                writer.writerow([
                    f"{t:.4f}",
                    *xR_meas.tolist(), *xL_meas.tolist(),
                    *xR_d.tolist(), *xL_d.tolist(),
                ])
                time.sleep(dt)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # === 8. Helpers ===
    def set_ref(arm, x, R):
        key = "R" if arm == "R" else "L"
        with ref_lock:
            ref[key]["x"] = np.asarray(x, float).copy()
            ref[key]["R"] = np.asarray(R, float).copy()

    def get_meas(arm):
        return robot.get_R_ee_pose() if arm == "R" else robot.get_L_ee_pose()

    def wait_converged(arm, timeout_s=SETTLE_TIMEOUT_S, tol=POS_TOL_M):
        t_start = time.time()
        with ref_lock:
            x_target = ref["R" if arm == "R" else "L"]["x"].copy()
        while time.time() - t_start < timeout_s:
            x_meas, _ = get_meas(arm)
            err = np.linalg.norm(x_target - x_meas)
            if err < tol:
                print(f"    [settle] arm={arm} converged, err={err*100:.2f} cm")
                return True
            time.sleep(0.1)
        x_meas, _ = get_meas(arm)
        err = np.linalg.norm(x_target - x_meas)
        print(f"    [settle] arm={arm} TIMEOUT after {timeout_s}s, err={err*100:.2f} cm")
        return False

    def go_home(arm):
        x_zero = xR_zero if arm == "R" else xL_zero
        R_zero = RR_zero if arm == "R" else RL_zero
        set_ref(arm, x_zero, R_zero)
        wait_converged(arm)

    # === 9. Home both arms before starting the orientation sweeps ===
    print("[INFO] Homing both arms to zero pose...")
    try:
        while True:
            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()
            err_R = np.linalg.norm(xR_zero - xR_meas)
            err_L = np.linalg.norm(xL_zero - xL_meas)
            if err_R < POS_TOL_M and err_L < POS_TOL_M:
                print(f"[INFO] Reached zero pose (err_R={err_R:.3f}, err_L={err_L:.3f} m).")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted while homing.")
        run_flag["running"] = False
        ctrl_thread.join(timeout=1.0)
        robot.stop()
        return

    # === 10. Orientation test sequence: one arm at a time, one mode at a time ===
    def run_arm_mode(arm, mode):
        print(f"\n===== ARM={arm}  MODE={mode} =====")
        go_home(arm)
        pause()

        for name, offset in TEST_OFFSETS.items():
            x_zero = xR_zero if arm == "R" else xL_zero
            x_target = x_zero + offset
            R_target = compute_grasp_orientation(arm, x_target, mode=mode)

            print(f"\n  -- target: {name}  x_target={np.round(x_target, 4)}")
            print(f"     R_target=\n{np.round(R_target, 4)}")
            pause(f"     About to command arm {arm}. Press Enter to send...")

            set_ref(arm, x_target, R_target)
            wait_converged(arm)

            print(f"     Returning arm {arm} to zero pose.")
            go_home(arm)
            pause()

        print(f"===== DONE: ARM={arm}  MODE={mode} =====")

    try:
        for arm in ("R", "L"):
            for mode in ("palm_down", "inward"):
                run_arm_mode(arm, mode)
        print("\n[INFO] All grasp-orientation tests complete.")
    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user during orientation tests.")

    # === 11. Shutdown ===
    print("[INFO] Stopping control thread and robot...")
    run_flag["running"] = False
    ctrl_thread.join(timeout=2.0)
    robot.stop()
    print("[INFO] Done. Log saved to:", log_path)


if __name__ == "__main__":
    main()
