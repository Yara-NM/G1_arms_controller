"""
test_interactive_profiles.py
----------------------------
Interactive task-space test for calibrating impedance/speed PROFILES on the G1.

Flow you asked for:
  1. Boots the one-layer PD controller + a SpringDamperCartesian per arm.
  2. Homes both arms to the "zero" pose.
  3. Drops you into a menu where you can:
        p : change PROFILE  (choose arm, choose soft/medium/stiff)
        m : MOVE reference  (choose arm, axis, +/- cm)
        i : INFO            (current profile + refs + tracking error)
        z : re-home to zero pose
        q : quit
  4. A background thread runs spring -> IK -> joint PD at 50 Hz and logs to CSV.

Switching a profile changes only K/D + ramp speed IN PLACE, so the arm keeps
its pose and you immediately feel the new behavior when you next push it or jog it.

Requires:
  - spring_damper_cartesian.py with set_gains() + set_ref_speed()  (the patched version)
  - profiles.py in the same package (task_space_impedance/)
"""

import time
import os
import threading
import csv
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian
from task_space_impedance.profiles import PROFILES, apply_profile, describe, DEFAULT_PROFILE

# ----------------------------------------------------------------------------
# Config -- flip these when moving from sim to the real robot
# ----------------------------------------------------------------------------
DDS_DOMAIN = 0          # 1 = sim/loopback, 0 = real robot
DDS_IFACE  = "enp2s0"       # "lo" for sim, "enp2s0" for real robot
CTRL_DT    = 0.02       # 50 Hz joint + spring-damper loop
SPEED_SCALE = 1.0       # global multiplier on profile ramp speed (1.0 / 0.9 / 0.7 ...)
ZERO_ERR_THRESHOLD = 0.05   # [m] "reached zero" tolerance


def get_zero_poses():
    """
    Chosen 'zero' EE poses for right and left arms
    (elbows-folded, hands-forward, with small offsets).
    """
    xR_zero = np.array([0.32686 , -0.15184 , 0.06149])
    RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                        [-0.01149,  0.99992, -0.00482],
                        [-0.11184,  0.00350,  0.99372]])

    xL_zero = np.array([0.32683 ,  0.15156 , 0.06145])
    RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                        [ 0.01129,  0.99992,  0.00597],
                        [-0.11195, -0.00467,  0.99370]])

    return xR_zero, RR_zero, xL_zero, RL_zero


def ask_arm(prompt="Arm [r/l/b]: "):
    """Return 'r', 'l', 'b', or None if invalid."""
    a = input(prompt).strip().lower()
    if a in ("r", "l", "b"):
        return a
    print("[WARN] Invalid arm (use r / l / b).")
    return None


def main():
    # === 0. DDS init ===
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS initialized (domain={DDS_DOMAIN}, iface={DDS_IFACE}).")
    except Exception as e:
        print("[WARN] DDS init issue:", e)

    # === 1. Init robot controller: ONE-LAYER PD ONLY (no torque/impedance pipeline) ===
    robot = G1RobotArmController(
        ctrl_dt=CTRL_DT,
        ctrl_2l_dt=None,   # force one-layer controller
        mode='l',          # low-level
        visualize=False,
        imp=False
    )
    robot.start()
    time.sleep(5.0)  # let encoders & DDS settle

    # === 2. Read current EE poses (start pose) ===
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()
    print("[INFO] Start pose:")
    print("  Right pos:", xR_start)
    print("  Left  pos:", xL_start)

    # === 3. Zero / home EE poses ===
    xR_ref, RR_ref, xL_ref, RL_ref = get_zero_poses()

    # === 4. Spring-damper modules (gains come from the DEFAULT profile) ===
    spring_R = SpringDamperCartesian()
    spring_L = SpringDamperCartesian()

    profile_of = {"R": DEFAULT_PROFILE, "L": DEFAULT_PROFILE}
    lbl_R = apply_profile(spring_R, DEFAULT_PROFILE, speed_scale=SPEED_SCALE)
    lbl_L = apply_profile(spring_L, DEFAULT_PROFILE, speed_scale=SPEED_SCALE)
    print(f"[INFO] Default profile applied to both arms: {lbl_R}")

    # 1) initial reference ("zero") pose per arm
    spring_R.set_target(xR_ref, RR_ref)
    spring_L.set_target(xL_ref, RL_ref)
    # 2) reset offsets so we start from the *current* measured pose
    spring_R.reset(xR_start, RR_start)
    spring_L.reset(xL_start, RL_start)

    # === 5. Shared state for threads ===
    run_flag = {"running": True}
    ref_lock = threading.Lock()

    xR_ref_current = xR_ref.copy()
    xL_ref_current = xL_ref.copy()
    RR_ref_current = RR_ref.copy()
    RL_ref_current = RL_ref.copy()

    # === 6. Logging setup ===
    log_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "task_space_profiles_log.csv")
    print(f"[INFO] Logging EE positions to: {log_path}")

    # === 7. Control loop thread ===
    def control_loop():
        nonlocal xR_ref_current, xL_ref_current, RR_ref_current, RL_ref_current
        t0 = time.time()
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t", "prof_R", "prof_L",
                "xR_meas_x", "xR_meas_y", "xR_meas_z",
                "xL_meas_x", "xL_meas_y", "xL_meas_z",
                "xR_des_x",  "xR_des_y",  "xR_des_z",
                "xL_des_x",  "xL_des_y",  "xL_des_z",
            ])

            while run_flag["running"]:
                dt = CTRL_DT
                t = time.time() - t0

                with ref_lock:
                    spring_R.set_target(xR_ref_current, RR_ref_current)
                    spring_L.set_target(xL_ref_current, RL_ref_current)
                    pR, pL = profile_of["R"], profile_of["L"]

                    xR_d, RR_d, _ = spring_R.step(dt)
                    xL_d, RL_d, _ = spring_L.step(dt)

                # send desired pose through IK + joint PD
                robot.move_arms_with_Rt(
                    left_R=RL_d,  left_t=xL_d,
                    right_R=RR_d, right_t=xR_d
                )

                # measure for logging
                xR_meas, _ = robot.get_R_ee_pose()
                xL_meas, _ = robot.get_L_ee_pose()
                writer.writerow([
                    f"{t:.4f}", pR, pL,
                    *xR_meas.tolist(), *xL_meas.tolist(),
                    *xR_d.tolist(),    *xL_d.tolist(),
                ])

                time.sleep(dt)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # === 8. Wait until we're basically at zero pose ===
    print("[INFO] Moving to zero pose...")
    try:
        while True:
            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()
            err_R = np.linalg.norm(xR_ref_current - xR_meas)
            err_L = np.linalg.norm(xL_ref_current - xL_meas)
            if err_R < ZERO_ERR_THRESHOLD and err_L < ZERO_ERR_THRESHOLD:
                print(f"[INFO] Reached zero pose (err_R={err_R:.3f}, err_L={err_L:.3f} m).")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted while homing.")
        run_flag["running"] = False
        ctrl_thread.join(timeout=1.0)
        robot.stop()
        return

    # === 9. Interactive menu ===
    def print_menu():
        print("\n=== COMMANDS ===")
        print("  p : change PROFILE (arm -> soft/medium/stiff)")
        print("  m : MOVE reference (arm -> axis -> delta cm)")
        print("  i : INFO (profiles, refs, tracking error)")
        print("  z : re-home to zero pose")
        print("  q : quit")
        print("  profiles available:")
        for name in PROFILES:
            print("     -", describe(name))

    def do_profile():
        arm = ask_arm("Apply profile to arm [r/l/b]: ")
        if arm is None:
            return
        name = input(f"Profile {list(PROFILES)}: ").strip().lower()
        if name not in PROFILES:
            print("[WARN] Unknown profile.")
            return
        with ref_lock:
            if arm in ("r", "b"):
                lbl = apply_profile(spring_R, name, speed_scale=SPEED_SCALE)
                profile_of["R"] = name
                print(f"[CMD] Right -> {lbl}")
            if arm in ("l", "b"):
                lbl = apply_profile(spring_L, name, speed_scale=SPEED_SCALE)
                profile_of["L"] = name
                print(f"[CMD] Left  -> {lbl}")

    def do_move():
        nonlocal xR_ref_current, xL_ref_current
        arm = ask_arm("Move arm [r/l/b]: ")
        if arm is None:
            return
        axis = input("Axis [x/y/z]: ").strip().lower()
        if axis not in ("x", "y", "z"):
            print("[WARN] Invalid axis.")
            return
        try:
            delta_cm = float(input("Delta (cm, e.g. 5 or -10): ").strip())
        except ValueError:
            print("[WARN] Could not parse number.")
            return
        d = np.zeros(3)
        d[{"x": 0, "y": 1, "z": 2}[axis]] = 0.01 * delta_cm
        with ref_lock:
            if arm in ("r", "b"):
                xR_ref_current = xR_ref_current + d
                print(f"[CMD] New right ref: {xR_ref_current}")
            if arm in ("l", "b"):
                xL_ref_current = xL_ref_current + d
                print(f"[CMD] New left  ref: {xL_ref_current}")

    def do_info():
        xR_meas, _ = robot.get_R_ee_pose()
        xL_meas, _ = robot.get_L_ee_pose()
        with ref_lock:
            err_R = np.linalg.norm(xR_ref_current - xR_meas)
            err_L = np.linalg.norm(xL_ref_current - xL_meas)
            print(f"\n[INFO] Right | profile={profile_of['R']:6s} "
                  f"| ref={np.round(xR_ref_current,3)} meas={np.round(xR_meas,3)} "
                  f"| err={err_R*100:.2f} cm")
            print(f"[INFO] Left  | profile={profile_of['L']:6s} "
                  f"| ref={np.round(xL_ref_current,3)} meas={np.round(xL_meas,3)} "
                  f"| err={err_L*100:.2f} cm")

    def do_rehome():
        nonlocal xR_ref_current, xL_ref_current, RR_ref_current, RL_ref_current
        xR_z, RR_z, xL_z, RL_z = get_zero_poses()
        with ref_lock:
            xR_ref_current = xR_z.copy(); RR_ref_current = RR_z.copy()
            xL_ref_current = xL_z.copy(); RL_ref_current = RL_z.copy()
        print("[CMD] Re-homing references to zero pose.")

    print("\n[INFO] Ready. You can now calibrate profiles interactively.")
    print_menu()

    try:
        while True:
            cmd = input("\ncmd [p/m/i/z/q]: ").strip().lower()
            if cmd == "q":
                print("[INFO] Quitting.")
                break
            elif cmd == "p":
                do_profile()
            elif cmd == "m":
                do_move()
            elif cmd == "i":
                do_info()
            elif cmd == "z":
                do_rehome()
            else:
                print_menu()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt in command loop.")

    # === 10. Shutdown ===
    print("[INFO] Stopping control thread and robot...")
    run_flag["running"] = False
    ctrl_thread.join(timeout=2.0)
    robot.stop()
    print("[INFO] Done. Log saved to:", log_path)


if __name__ == "__main__":
    main()
