import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian_old import SpringDamperCartesian
from scipy.spatial.transform import Rotation as R



if __name__ == "__main__":
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
        imp=False          # IMPORTANT: NO impedance/torque mapping here
    )

    # Start inner joint loop
    robot.start()
    time.sleep(1.0)  # let encoders & DDS settle

    # === 2. Read current EE poses (start pose) ===
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()

    print("[INFO] Start pose:")
    print("  Right pos:", xR_start)
    print("  Left  pos:", xL_start)

    # === 3. Define your ZERO / HOME EE poses ===
    # These are the *target* zero poses you gave (elbows folded, hands forward)
    xR_zero = np.array([0.32686 - 0.15, -0.15184 - 0.10, 0.06149])
    RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                        [-0.01149,  0.99992, -0.00482],
                        [-0.11184,  0.00350,  0.99372]])

    xL_zero = np.array([0.32683 - 0.15,  0.15156 + 0.10, 0.06145])
    RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                        [ 0.01129,  0.99992,  0.00597],
                        [-0.11195, -0.00467,  0.99370]])

    print("[INFO] Zero pose (target):")
    print("  Right pos:", xR_zero)
    print("  Left  pos:", xL_zero)

    # === 4. Spring–damper modules (position + orientation) ===
    # Gentle gains – you can tune later.
    spring_R = SpringDamperCartesian(
        K_p=(15.0, 15.0, 12.0),
        D_p=(2.0, 2.0, 2.0),
        M_p=(1.5, 1.5, 1.5),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2)
    )

    spring_L = SpringDamperCartesian(
        K_p=(15.0, 15.0, 12.0),
        D_p=(2.0, 2.0, 2.0),
        M_p=(1.5, 1.5, 1.5),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2)
    )

    # 1) Set reference ("zero") pose for each arm
    spring_R.set_target(xR_zero, RR_zero)
    spring_L.set_target(xL_zero, RL_zero)

    # 2) Reset offsets so we start from the *current* measured pose
    spring_R.reset(xR_start, RR_start)
    spring_L.reset(xL_start, RL_start)

    print("[INFO] Spring–damper to ZERO pose (position + orientation).")
    print("[INFO] Arms should move smoothly from current pose to zero pose.\n")

    # === 5. Main loop ===
    t0 = time.time()
    last_print_sec = -1

    try:
        while True:
            t = time.time() - t0
            dt = ctrl_dt

            # --- spring–damper step: get desired EE pose ---
            xR_d, RR_d, dbg_R = spring_R.step(dt)
            xL_d, RL_d, dbg_L = spring_L.step(dt)

            # --- send desired pose through IK + joint PD ---
            robot.move_arms_with_Rt(
                left_R=RL_d,  left_t=xL_d,
                right_R=RR_d, right_t=xR_d
            )

            # --- diagnostics every ~1 second ---
            sec = int(t)
            if sec != last_print_sec:
                last_print_sec = sec
                err_R = np.linalg.norm(xR_zero - xR_d)
                err_L = np.linalg.norm(xL_zero - xL_d)
                print(f"[INFO] t = {t:4.1f} s | "
                      f"‖e_R‖ = {err_R:.4f} m, ‖e_L‖ = {err_L:.4f} m")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping spring–damper go-to-zero test…")
    finally:
        robot.stop()
        print("[INFO] Robot stopped. Demo complete.")
