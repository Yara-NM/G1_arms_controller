#!/usr/bin/env python3
"""
minimal_zero_test.py — the simplest possible motion test.

Robot initializes, homes to the zero pose (slowly), holds briefly, then the
controller stops and the program exits. Nothing else: no points, no camera,
no targets other than zero, no interactive prompts.

Use this to watch how the arms move for the first time. Uses the SOFT profile
(lowest stiffness, slowest ramp) so the motion is gentle and easy to observe.

If THIS moves cleanly to zero, the controller stack is fine and the earlier
"went crazy" was the specific target/profile (a low hanging pose at higher
stiffness), not the controller itself.
"""

import time
import threading

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from arms_controller.g1_robot_controller import G1RobotArmController
from task_space_impedance.spring_damper_cartesian import SpringDamperCartesian
from task_space_impedance.profiles import apply_profile, describe

# ============================================================================
# CONFIG
# ============================================================================
SIM         = True                 # True = MuJoCo/loopback, False = real robot
DDS_DOMAIN  = 1 if SIM else 0
DDS_IFACE   = "lo" if SIM else "enp2s0"
CTRL_DT     = 0.02                 # 50 Hz control loop
PROFILE     = "soft"               # gentlest/slowest for a first-ever motion
SETTLE_START_S  = 5.0              # wait after start() for encoders/DDS to settle
HOME_WATCH_S    = 12.0             # how long to watch it home (prints error/sec)
HOLD_AT_ZERO_S  = 5.0              # sit at zero before stopping
# ============================================================================


def get_zero_poses():
    """Safe home EE poses (same values used across the pipeline)."""
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
    # --- DDS + one-layer controller ---
    try:
        ChannelFactoryInitialize(DDS_DOMAIN, DDS_IFACE)
        time.sleep(0.5)
        print(f"[INFO] DDS init (domain={DDS_DOMAIN}, iface='{DDS_IFACE}').")
    except Exception as e:
        print("[WARN] DDS init:", e)

    robot = G1RobotArmController(ctrl_dt=CTRL_DT, ctrl_2l_dt=None,
                                 mode='l', visualize=False, imp=False)
    robot.start()
    print(f"[INFO] Started. Settling for {SETTLE_START_S}s...")
    time.sleep(SETTLE_START_S)

    # --- read current pose, set up spring toward zero ---
    xR_start, RR_start = robot.get_R_ee_pose()
    xL_start, RL_start = robot.get_L_ee_pose()
    xR_zero, RR_zero, xL_zero, RL_zero = get_zero_poses()

    print(f"[INFO] Start pose  R={np.round(xR_start,3)}  L={np.round(xL_start,3)}")
    print(f"[INFO] Zero target R={np.round(xR_zero,3)}  L={np.round(xL_zero,3)}")

    spring_R = SpringDamperCartesian()
    spring_L = SpringDamperCartesian()
    apply_profile(spring_R, PROFILE)
    apply_profile(spring_L, PROFILE)
    print(f"[INFO] Profile: {describe(PROFILE)}")

    spring_R.set_target(xR_zero, RR_zero); spring_R.reset(xR_start, RR_start)
    spring_L.set_target(xL_zero, RL_zero); spring_L.reset(xL_start, RL_start)

    CTRL_RUN = {"on": True}

    def control_loop():
        # Target is fixed (zero) for the whole test -> no set_target in loop,
        # no shared refs, no locks. Just: step -> send. As lean as it gets.
        while CTRL_RUN["on"]:
            xR_d, RR_d, _ = spring_R.step(CTRL_DT)
            xL_d, RL_d, _ = spring_L.step(CTRL_DT)
            robot.move_arms_with_Rt(left_R=RL_d, left_t=xL_d,
                                    right_R=RR_d, right_t=xR_d)
            time.sleep(CTRL_DT)

    ctrl_t = threading.Thread(target=control_loop, daemon=True)
    ctrl_t.start()

    # --- watch it home (prints error each second) ---
    print("\n[INFO] Homing to zero — watch the arms...")
    t0 = time.time()
    while time.time() - t0 < HOME_WATCH_S:
        xR_m, _ = robot.get_R_ee_pose()
        xL_m, _ = robot.get_L_ee_pose()
        eR = np.linalg.norm(xR_zero - xR_m)
        eL = np.linalg.norm(xL_zero - xL_m)
        print(f"  t={time.time()-t0:4.1f}s  errR={eR*100:5.2f}cm  errL={eL*100:5.2f}cm")
        if eR < 0.02 and eL < 0.02:
            print("[INFO] Reached zero.")
            break
        time.sleep(1.0)

    # --- hold at zero, then stop cleanly ---
    print(f"[INFO] Holding at zero for {HOLD_AT_ZERO_S}s...")
    time.sleep(HOLD_AT_ZERO_S)

    print("[INFO] Stopping controller...")
    CTRL_RUN["on"] = False          # stop the loop while the arm is static at zero
    ctrl_t.join(timeout=2.0)
    robot.stop()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()