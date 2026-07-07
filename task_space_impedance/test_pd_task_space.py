import time
import os
import csv
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from scipy.spatial.transform import Rotation as R


LOG_PATH = os.path.join(os.path.dirname(__file__), "task_space_log_pd.csv")


def control_loop(robot,
                 ctrl_dt,
                 stop_event,
                 shared):
    """
    Background loop:
      - ramp x_cmd toward x_ref with a max velocity
      - send commanded pose via IK + PD
      - log measured & commanded EE positions
    """
    # Open CSV + header
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "xR_meas_x", "xR_meas_y", "xR_meas_z",
            "xL_meas_x", "xL_meas_y", "xL_meas_z",
            "xR_des_x",  "xR_des_y",  "xR_des_z",
            "xL_des_x",  "xL_des_y",  "xL_des_z",
        ])

        t0 = time.time()

        # max Cartesian speed for the commanded pose [m/s]
        v_max = 0.05  # you can tune this

        while not stop_event.is_set():
            t = time.time() - t0

            # --- read measured EE pose ---
            xR_meas, RR_meas = robot.get_R_ee_pose()
            xL_meas, RL_meas = robot.get_L_ee_pose()

            # --- ramp commanded positions toward reference ---
            # We only update shared["x*_cmd"]; x*_ref is set by main thread.
            for arm in ("R", "L"):
                x_cmd = shared[f"x{arm}_cmd"]
                x_ref = shared[f"x{arm}_ref"]

                dx = x_ref - x_cmd
                dist = np.linalg.norm(dx)

                if dist > 1e-6:
                    max_step = v_max * ctrl_dt
                    if dist > max_step:
                        dx = dx * (max_step / dist)
                    x_cmd = x_cmd + dx

                shared[f"x{arm}_cmd"] = x_cmd

            # Command orientations: keep fixed "zero" orientations
            RR_cmd = shared["RR_zero"]
            RL_cmd = shared["RL_zero"]

            # --- send to robot via IK + PD ---
            robot.move_arms_with_Rt(
                left_R=RL_cmd,  left_t=shared["xL_cmd"],
                right_R=RR_cmd, right_t=shared["xR_cmd"]
            )

            # --- log ---
            writer.writerow([
                t,
                xR_meas[0], xR_meas[1], xR_meas[2],
                xL_meas[0], xL_meas[1], xL_meas[2],
                shared["xR_cmd"][0], shared["xR_cmd"][1], shared["xR_cmd"][2],
                shared["xL_cmd"][0], shared["xL_cmd"][1], shared["xL_cmd"][2],
            ])

            f.flush()
            time.sleep(ctrl_dt)


def main():
    # === 0. DDS init ===
    try:
        ChannelFactoryInitialize(1, "lo")
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
        imp=False          # IMPORTANT: PD-only, no impedance / torque mapping
    )

    # Start PD joint loop
    robot.start()
    time.sleep(1.0)  # let encoders & DDS settle

    # === 2. Read current EE poses (start pose) ===
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()

    print("[INFO] Start pose:")
    print("  Right pos:", xR_start)
    print("  Left  pos:", xL_start)

    # === 3. Define ZERO / HOME EE pose (same as in impedance test) ===
    xR_zero = np.array([0.17686, -0.25184, 0.06149])
    RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                        [-0.01149,  0.99992, -0.00482],
                        [-0.11184,  0.00350,  0.99372]])

    xL_zero = np.array([0.17683,  0.25156, 0.06145])
    RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                        [ 0.01129,  0.99992,  0.00597],
                        [-0.11195, -0.00467,  0.99370]])

    print("[INFO] Zero pose (initial reference):")
    print("  Right pos:", xR_zero)
    print("  Left  pos:", xL_zero)

    # === 4. Shared state for control thread ===
    shared = {
        # current commanded EE positions (start from measured pose)
        "xR_cmd": xR_start.copy(),
        "xL_cmd": xL_start.copy(),
        # reference EE positions (we'll ramp toward these)
        "xR_ref": xR_zero.copy(),
        "xL_ref": xL_zero.copy(),
        # fixed zero orientations
        "RR_zero": RR_zero,
        "RL_zero": RL_zero,
    }

    print("[INFO] PD-only go-to-zero in task space (with ramped reference).")
    print("[INFO] Logging EE positions to:", LOG_PATH)

    # === 5. Start background control thread ===
    stop_event = threading.Event()
    th = threading.Thread(
        target=control_loop,
        args=(robot, ctrl_dt, stop_event, shared),
        daemon=True
    )
    th.start()

    # === 6. Wait until close to zero pose, then allow user commands ===
    print("[INFO] Moving to zero pose…")

    # Rough wait for convergence (you can improve this with a real check)
    time.sleep(6.0)

    print("[INFO] You can now command movements (PD-only planner).")
    print("      Commands:")
    print("        arm  : r / l / b (both) / q (quit)")
    print("        axis : x / y / z")
    print("        Δcm  : displacement in centimeters (e.g. 5 or -10)\n")

    try:
        while True:
            arm = input("Arm to move [r/l/b/q]: ").strip().lower()
            if arm == "q":
                print("[INFO] Quitting interactive loop.")
                break
            if arm not in ("r", "l", "b"):
                print("[WARN] Invalid arm; use r / l / b / q.")
                continue

            axis = input("Axis to move [x/y/z]: ").strip().lower()
            if axis not in ("x", "y", "z"):
                print("[WARN] Invalid axis; use x / y / z.")
                continue
            idx = {"x": 0, "y": 1, "z": 2}[axis]

            try:
                d_cm = float(input("Delta (cm, e.g. 5 or -10): ").strip())
            except ValueError:
                print("[WARN] Invalid number.")
                continue

            d_m = d_cm / 100.0

            # Update reference positions
            if arm in ("r", "b"):
                shared["xR_ref"] = shared["xR_ref"].copy()
                shared["xR_ref"][idx] += d_m
            if arm in ("l", "b"):
                shared["xL_ref"] = shared["xL_ref"].copy()
                shared["xL_ref"][idx] += d_m

            print("[CMD] New right ref pos:", shared["xR_ref"])
            print("[CMD] New left ref pos: ", shared["xL_ref"])

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # Stop thread and robot
        stop_event.set()
        th.join(timeout=2.0)
        robot.stop()
        print("[INFO] Robot stopped. PD-only demo complete.")
        print("[INFO] Log saved to:", LOG_PATH)


if __name__ == "__main__":
    main()
