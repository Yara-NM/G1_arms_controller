import time
import os
import threading
import csv
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian
# from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian

def get_zero_poses():
    """
    Return your chosen 'zero' EE poses for right and left arms.
    These are the elbows-folded, hands-forward poses (with small offsets).
    """
    xR_zero = np.array([0.32686 - 0.15, -0.15184 - 0.10, 0.06149])
    RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                        [-0.01149,  0.99992, -0.00482],
                        [-0.11184,  0.00350,  0.99372]])

    xL_zero = np.array([0.32683 - 0.15,  0.15156 + 0.10, 0.06145])
    RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                        [ 0.01129,  0.99992,  0.00597],
                        [-0.11195, -0.00467,  0.99370]])

    return xR_zero, RR_zero, xL_zero, RL_zero


def main():
    # === 0. DDS init ===
    try:
        ChannelFactoryInitialize(1, "lo")
        # ChannelFactoryInitialize(0, "enp2s0")
        
        time.sleep(0.5)
        print("[INFO] DDS initialized.")
    except Exception as e:
        print("[WARN] DDS init issue:", e)

    # === 1. Init robot controller: ONE-LAYER PD ONLY ===
    ctrl_dt = 0.02  # 50 Hz joint loop
    robot = G1RobotArmController(
        ctrl_dt=ctrl_dt,
        ctrl_2l_dt=None,   # force one-layer controller
        mode='l',          # low-level
        visualize=False,
        imp=False          # IMPORTANT: NO torque/impedance pipeline
    )

    # Start inner joint loop
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

    print("[INFO] Zero pose (initial reference):")
    print("  Right pos:", xR_ref)
    print("  Left  pos:", xL_ref)

    # === 4. Spring–damper modules (position + orientation) ===
    # Gentle gains – tune later if you want.

    K = (5.0,5.0 , 3.0 ) 
    D = (3.0, 3.0, 2.0)
    spring_R = SpringDamperCartesian(
        K_p= K,
        D_p= D,
        M_p=(1.5, 1.5, 1.5),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2)
    )

    spring_L = SpringDamperCartesian(
        K_p= K,
        D_p= D,
        M_p=(1.5, 1.5, 1.5),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2)
    )

    # 1) Set initial reference ("zero") pose for each arm
    spring_R.set_target(xR_ref, RR_ref)
    spring_L.set_target(xL_ref, RL_ref)

    # 2) Reset offsets so we start from the *current* measured pose
    spring_R.reset(xR_start, RR_start)
    spring_L.reset(xL_start, RL_start)

    print("[INFO] Spring–damper to ZERO pose (position + orientation).")
    print("[INFO] A background thread will run the control;")
    print("       You will be able to interactively move the arms in x/y/z.\n")

    # === 5. Shared state for threads ===
    run_flag = {"running": True}        # simple mutable flag
    ref_lock = threading.Lock()         # protect updates of references

    # We keep references as mutable numpy arrays so we can update them
    # in the main thread when you type commands.
    xR_ref_current = xR_ref.copy()
    xL_ref_current = xL_ref.copy()
    RR_ref_current = RR_ref.copy()
    RL_ref_current = RL_ref.copy()

    # === 6. Logging setup ===
    log_dir = os.path.dirname(__file__)
    log_path = os.path.join(log_dir,"results/", "task_space_log_2_real.csv")
    print(f"[INFO] Logging EE positions to: {log_path}")

    # === 7. Control loop thread ===
    def control_loop():
        nonlocal xR_ref_current, xL_ref_current, RR_ref_current, RL_ref_current
        t0 = time.time()

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "xR_meas_x", "xR_meas_y", "xR_meas_z",
                "xL_meas_x", "xL_meas_y", "xL_meas_z",
                "xR_des_x",  "xR_des_y",  "xR_des_z",
                "xL_des_x",  "xL_des_y",  "xL_des_z"
            ])

            last_print_sec = -1

            while run_flag["running"]:
                now = time.time()
                t = now - t0
                dt = ctrl_dt

                # --- spring–damper step: compute desired EE poses ---
                with ref_lock:
                    # ensure springs use the latest targets
                    spring_R.set_target(xR_ref_current, RR_ref_current)
                    spring_L.set_target(xL_ref_current, RL_ref_current)

                xR_d, RR_d, dbg_R = spring_R.step(dt)
                xL_d, RL_d, dbg_L = spring_L.step(dt)

                # --- send desired pose through IK + joint PD ---
                robot.move_arms_with_Rt(
                    left_R=RL_d,  left_t=xL_d,
                    right_R=RR_d, right_t=xR_d
                )

                # --- measure current EE poses for logging ---
                xR_meas, _ = robot.get_R_ee_pose()
                xL_meas, _ = robot.get_L_ee_pose()

                writer.writerow([
                    f"{t:.4f}",
                    *xR_meas.tolist(),
                    *xL_meas.tolist(),
                    *xR_d.tolist(),
                    *xL_d.tolist(),
                ])

                # simple “are we at ref?” info every 1s
                sec = int(t)
                if sec != last_print_sec:
                    last_print_sec = sec
                    err_R = np.linalg.norm(xR_ref_current - xR_meas)
                    err_L = np.linalg.norm(xL_ref_current - xL_meas)
                    # print(f"[CTRL] t = {t:4.1f} s | "
                    #       f"‖e_R‖ = {err_R:.4f} m, ‖e_L‖ = {err_L:.4f} m")

                time.sleep(dt)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # === 8. Phase 1: wait until we’re basically at zero pose ===
    print("[INFO] Moving to zero pose…")
    try:
        while True:
            xR_meas, _ = robot.get_R_ee_pose()
            xL_meas, _ = robot.get_L_ee_pose()
            err_R = np.linalg.norm(xR_ref_current - xR_meas)
            err_L = np.linalg.norm(xL_ref_current - xL_meas)
            if err_R < 0.05 and err_L < 0.05:  # 2 cm threshold
                print(f"[INFO] Reached zero pose (within ~{err_L} cm).")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted while going to zero.")
        run_flag["running"] = False
        ctrl_thread.join(timeout=1.0)
        robot.stop()
        return

    print("\n[INFO] You can now command movements.")
    print("      Commands:")
    print("        arm  : r / l / b (both) / q (quit)")
    print("        axis : x / y / z")
    print("        Δcm  : displacement in centimeters (e.g. 5 or -10)\n")

    # === 9. Main thread: interactive command loop ===
    try:
        while True:
            arm = input("Arm to move [r/l/b/q]: ").strip().lower()
            if arm == "q":
                print("[INFO] Quitting interactive loop.")
                break
            if arm not in ("r", "l", "b"):
                print("[WARN] Invalid arm, please use r, l, b or q.")
                continue

            axis = input("Axis to move [x/y/z]: ").strip().lower()
            if axis not in ("x", "y", "z"):
                print("[WARN] Invalid axis, please use x, y or z.")
                continue

            try:
                delta_cm_str = input("Delta (cm, e.g. 5 or -10): ").strip()
                delta_cm = float(delta_cm_str)
            except ValueError:
                print("[WARN] Could not parse number, try again.")
                continue

            delta_m = 0.01 * delta_cm
            d = np.zeros(3)
            if axis == "x":
                d[0] = delta_m
            elif axis == "y":
                d[1] = delta_m
            else:  # "z"
                d[2] = delta_m

            with ref_lock:
                if arm in ("r", "b"):
                    xR_ref_current = xR_ref_current + d
                    print(f"[CMD] New right ref pos: {xR_ref_current}")
                if arm in ("l", "b"):
                    xL_ref_current = xL_ref_current + d
                    print(f"[CMD] New left ref pos:  {xL_ref_current}")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt in command loop.")

    # === 10. Shutdown ===
    print("[INFO] Stopping control thread and robot…")
    run_flag["running"] = False
    ctrl_thread.join(timeout=2.0)
    robot.stop()
    print("[INFO] Done. Log saved to:", log_path)


if __name__ == "__main__":
    main()
