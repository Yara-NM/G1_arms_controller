"""
grasp_orientation.py

Pure-math module for generating a feasible end-effector rotation (R_target)
for grasping a 3D point, given a fixed 'zero pose' and a chosen grasp mode
(palm_down or inward). No robot / controller dependency -- safe to run and
sanity-check standalone before wiring into the real pipeline.

DOF budget (why the method is shaped this way):
  - Full orientation = 3 DOF.
  - Choosing palm_down or inward fixes the palm-normal DIRECTION -> 2 DOF.
  - What's left is 1 DOF: twist about that normal (wrist yaw).
  Here we don't just fix the palm-normal direction -- we fix the *entire*
  zero-pose-derived rotation via a +/-90 deg fold about the hand's local
  x-axis (per your instruction), which nails all 3 DOF. The "interpolation"
  you asked for is then layered back on top as:
    - yaw   : rotation about world z, driven by target azimuth
              (this is the legitimate free wrist-twist DOF for a
              roughly-vertical approach axis, i.e. palm_down mode)
    - tilt  : rotation about the hand's local y-axis, driven by target
              elevation. This is NOT free in the strict sense -- it's a
              deliberately relaxed secondary DOF. Keep tilt_gain small,
              or set it to 0 if it visibly fights the grasp constraint
              during live testing.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Zero pose (right / left arm), from arm calibration
# ---------------------------------------------------------------------------
xR_zero = np.array([0.32686, -0.15184, 0.06149])
RR_zero = np.array([[ 0.99366,  0.01195,  0.11179],
                     [-0.01149,  0.99992, -0.00482],
                     [-0.11184,  0.00350,  0.99372]])

xL_zero = np.array([0.32683,  0.15156,  0.06145])
RL_zero = np.array([[ 0.99365, -0.01188,  0.11188],
                     [ 0.01129,  0.99992,  0.00597],
                     [-0.11195, -0.00467,  0.99370]])

ZERO = {
    'R': dict(x=xR_zero, R=RR_zero),
    'L': dict(x=xL_zero, R=RL_zero),
}

# ---------------------------------------------------------------------------
# UNVERIFIED -- confirm on hardware, then update this.
# Sign of the 90 deg rotation about the hand's LOCAL x-axis that folds the
# zero-pose (inward) orientation into palm-down for each arm.
# Guess: mirrored hands -> opposite signs. If the first live test shows
# the palm facing UP instead of down, flip that arm's sign. If it's facing
# down but rotated backwards/forwards from what you expect, the fold should
# probably be applied as R_zero @ rot_x(...) vs rot_x(...) @ R_zero instead
# -- try the alternate order in that case (see compute_grasp_orientation).
# ---------------------------------------------------------------------------
PALM_DOWN_SIGN = {'R': -1.0, 'L': +1.0}

# Secondary-DOF gains/limits (radians)
YAW_GAIN, MAX_YAW = 0.5, 0.35    # ~20 deg cap -- legitimate free wrist-twist
TILT_GAIN, MAX_TILT = 0.3, 0.25  # ~14 deg cap -- relaxed, disable if it fights the fold


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])


def compute_grasp_orientation(arm, x_target, mode='palm_down',
                               yaw_gain=YAW_GAIN, tilt_gain=TILT_GAIN,
                               max_yaw=MAX_YAW, max_tilt=MAX_TILT,
                               fold_order='post'):
    """
    Build a target end-effector rotation matrix for grasping x_target.

    arm  : 'R' or 'L'
    mode : 'palm_down' or 'inward'
    fold_order : 'post' -> R_mode = R_zero @ rot_x(sign*90deg)  (fold in hand-local frame)
                 'pre'  -> R_mode = rot_x(sign*90deg) @ R_zero  (fold in world frame)
                 Try 'post' first; switch to 'pre' if the observed rotation
                 direction doesn't match expectation on hardware.

    Returns a 3x3 rotation matrix (world frame, robot pelvis base).
    """
    if arm not in ('R', 'L'):
        raise ValueError(f"arm must be 'R' or 'L', got {arm!r}")

    x_zero = ZERO[arm]['x']
    R_zero = ZERO[arm]['R']

    # 1. fixed grasp-mode rotation
    if mode == 'palm_down':
        sign = PALM_DOWN_SIGN[arm]
        fold = rot_x(sign * np.pi / 2)
        R_mode = (R_zero @ fold) if fold_order == 'post' else (fold @ R_zero)
    elif mode == 'inward':
        R_mode = R_zero.copy()
    else:
        raise ValueError(f"unknown grasp mode: {mode!r} (use 'palm_down' or 'inward')")

    # 2. secondary rotation from target direction (relative to zero position)
    delta = np.asarray(x_target, dtype=float) - x_zero
    horiz_norm = float(np.hypot(delta[0], delta[1]))
    azimuth = np.arctan2(delta[1], delta[0]) if horiz_norm > 1e-6 else 0.0
    elevation = np.arctan2(delta[2], horiz_norm + 1e-9)

    yaw = float(np.clip(yaw_gain * azimuth, -max_yaw, max_yaw))
    tilt = float(np.clip(tilt_gain * elevation, -max_tilt, max_tilt))

    R_target = rot_z(yaw) @ R_mode @ rot_y(tilt)
    return R_target


def _is_valid_rotation(R, tol=1e-4):
    # tol=1e-4 because the zero-pose matrices above are given rounded to
    # 5 decimals, which alone produces ~1e-5-1e-6 orthonormality error.
    return (np.allclose(R.T @ R, np.eye(3), atol=tol) and
            abs(np.linalg.det(R) - 1.0) < tol)


if __name__ == "__main__":
    # Standalone sanity check -- no robot needed.
    sample_offsets = {
        'x+10cm': np.array([0.10, 0.0, 0.0]),
        'x-10cm': np.array([-0.10, 0.0, 0.0]),
        'y+10cm': np.array([0.0, 0.10, 0.0]),
        'y-10cm': np.array([0.0, -0.10, 0.0]),
        'z+10cm': np.array([0.0, 0.0, 0.10]),
        'z-10cm': np.array([0.0, 0.0, -0.10]),
        'x+20cm,z-10cm': np.array([0.20, 0.0, -0.10]),
    }

    print("Zero-pose matrix validity check:")
    print(f"  RR_zero valid rotation: {_is_valid_rotation(RR_zero)}")
    print(f"  RL_zero valid rotation: {_is_valid_rotation(RL_zero)}")
    print()

    for arm in ('R', 'L'):
        for mode in ('palm_down', 'inward'):
            print(f"--- arm={arm} mode={mode} ---")
            for name, offset in sample_offsets.items():
                x_target = ZERO[arm]['x'] + offset
                R = compute_grasp_orientation(arm, x_target, mode=mode)
                ok = _is_valid_rotation(R)
                print(f"  {name:16s} valid={ok}")
                if not ok:
                    print(f"    !! INVALID ROTATION MATRIX for {arm}/{mode}/{name}")
            print()
