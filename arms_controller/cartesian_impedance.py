
# cartesian_impedance.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def _to_vec(v, n):
    """
    Normalize a scalar / list / array into a float numpy vector of length n.
    - If v is scalar → broadcast to length n.
    - If v has length n → used as-is.
    """
    v = np.array(v, dtype=float).reshape(-1)
    if v.size == 1:
        v = np.repeat(v, n)
    assert v.size == n, f"Expected {n} elements, got {v.size}"
    return v

def so3_log(R_err):
    """
    Compute the rotation-vector log of an SO(3) rotation matrix.
    Input:
        R_err: (3,3) numpy array, rotation matrix
    Output:
        r: (3,) numpy array, axis*angle representation (angle in radians)
    """
    rot = R.from_matrix(R_err)
    ang = rot.magnitude()
    if ang < 1e-12:
        return np.zeros(3)
    return rot.as_rotvec()  # axis * angle

class CartesianImpedance:
    """
    Task-space impedance in WORLD frame:

      F* = Md_pos * a_d + D_pos * (v_d - v) + K_pos * (x_d - x) + F_bias_pos
      M* = Md_ori * aW_d + D_ori * (w_d - w) + K_ori * e_ori + F_bias_ori

    where all gain vectors are treated as diagonals of 3×3 matrices.
    The final wrench is:
        wrench = [F_x, F_y, F_z, M_x, M_y, M_z]^T  (WORLD frame).
    """
    def __init__(self,
                 K_pos=(100,100,50), D_pos=(20,20,10), Md_pos=(2.0,2.0,1.0),
                 K_ori=(3,3,2),        D_ori=(0.5,0.5,0.4), Md_ori=(0.2,0.2,0.1),
                 Fmax=10.0, Mmax=5.0,
                 vel_alpha=0.2):
        # Translational
        self.K_pos  = _to_vec(K_pos, 3)
        self.D_pos  = _to_vec(D_pos, 3)
        self.Md_pos = _to_vec(Md_pos, 3)
        # Rotational
        self.K_ori  = _to_vec(K_ori, 3)
        self.D_ori  = _to_vec(D_ori, 3)
        self.Md_ori = _to_vec(Md_ori, 3)

        # Saturation
        self.Fmax = float(Fmax)
        self.Mmax = float(Mmax)

        # Velocity filter coefficient
        self.vel_alpha = float(vel_alpha)

        # Internal state
        self._x_prev = None
        self._R_prev = None
        self._v      = np.zeros(3)  # filtered linear velocity
        self._omega  = np.zeros(3)  # filtered angular velocity

    def reset(self):
        """
        Clear internal velocity/orientation filters.
        Use when starting a new task or after a large discontinuity.
        """
        self._x_prev = None
        self._R_prev = None
        self._v[:] = 0.0
        self._omega[:] = 0.0

    def set_params(self, K_pos=None, D_pos=None, Md_pos=None,
                   K_ori=None, D_ori=None, Md_ori=None):
        """
        Update any subset of the gains / inertias at runtime.
        Arguments are 3-vectors or scalars (broadcast).
        """
        if K_pos is not None:  self.K_pos  = _to_vec(K_pos,  3)
        if D_pos is not None:  self.D_pos  = _to_vec(D_pos,  3)
        if Md_pos is not None: self.Md_pos = _to_vec(Md_pos, 3)

        if K_ori is not None:  self.K_ori  = _to_vec(K_ori,  3)
        if D_ori is not None:  self.D_ori  = _to_vec(D_ori,  3)
        if Md_ori is not None: self.Md_ori = _to_vec(Md_ori, 3)

    def update(self, dt,
               x_d, R_d,              # desired pose (WORLD)
               x,   R_m,              # measured pose (WORLD)
               v_d=None,  w_d=None,   # desired linear/angular velocities (WORLD)
               a_d=None,  aW_d=None,  # desired linear/angular accelerations (WORLD)
               F_bias=None):
        """
        Compute the WORLD-frame wrench given desired & measured pose/twist.


        Args:
            dt      : float, control period [s]
            x_d     : (3,) desired position
            R_d     : (3,3) desired rotation matrix
            x       : (3,) measured position
            R_m     : (3,3) measured rotation matrix
            v_d     : (3,) desired linear velocity (default 0)
            w_d     : (3,) desired angular velocity (default 0)
            a_d     : (3,) desired linear acceleration (default 0)
            aW_d    : (3,) desired angular acceleration in WORLD (default 0)
            F_bias  : (6,) constant wrench bias [Fx,Fy,Fz,Mx,My,Mz] (default 0)

        Returns:
            wrench : (6,) numpy array, [Fx,Fy,Fz, Mx,My,Mz] in WORLD frame,
                     saturated to Fmax / Mmax.
        """
        # Defaults
        if v_d  is None: v_d  = np.zeros(3)
        if w_d  is None: w_d  = np.zeros(3)
        if a_d  is None: a_d  = np.zeros(3)
        if aW_d is None: aW_d = np.zeros(3)
        if F_bias is None: F_bias = np.zeros(6)

        x_d  = np.asarray(x_d, float).reshape(3)
        x    = np.asarray(x,   float).reshape(3)
        R_d  = np.asarray(R_d, float).reshape(3,3)
        R_m  = np.asarray(R_m, float).reshape(3,3)
        v_d  = np.asarray(v_d, float).reshape(3)
        w_d  = np.asarray(w_d, float).reshape(3)
        a_d  = np.asarray(a_d, float).reshape(3)
        aW_d = np.asarray(aW_d,float).reshape(3)
        F_bias = np.asarray(F_bias, float).reshape(6)

        # ---- 1) Position/orientation errors ----
        e_pos = x_d - x                         # (3,)
        R_err = R_d.T @ R_m                     # (3,3)
        e_ori = so3_log(R_err)                  # (3,)

        # ---- 2) Estimate linear & angular velocity via filtered finite diff ----
        if self._x_prev is not None:
            dt_eff = max(dt, 1e-3)

            # Linear velocity
            v_meas = (x - self._x_prev) / dt_eff
            self._v = (1.0 - self.vel_alpha) * self._v + self.vel_alpha * v_meas

            # Angular velocity from relative rotation
            dR = R_m @ self._R_prev.T
            omg_meas = so3_log(dR) / dt_eff
            self._omega = (1.0 - self.vel_alpha) * self._omega + self.vel_alpha * omg_meas

        # Update stored pose
        self._x_prev = x.copy()
        self._R_prev = R_m.copy()

        # ---- 3) Compute wrench ----
        # Translational part
        F = self.Md_pos * a_d \
          + self.D_pos * (v_d - self._v) \
          + self.K_pos * e_pos

        # Rotational part
        M = self.Md_ori * aW_d \
          + self.D_ori * (w_d - self._omega) \
          + self.K_ori * e_ori

        wrench = np.hstack([F, M]) + F_bias

        # ---- 4) Saturation ----
        wrench[:3] = np.clip(wrench[:3], -self.Fmax, self.Fmax)
        wrench[3:] = np.clip(wrench[3:], -self.Mmax, self.Mmax)

        # ---- 5) NaN / Inf guard ----
        if not np.all(np.isfinite(wrench)):
            wrench[:] = 0.0

        return wrench


# ---------------------------------------------------------------------------
# Minimal standalone tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple unit tests / demos for debugging

    ci = CartesianImpedance()

    dt = 0.01

    # Identity rotation
    R_I = np.eye(3)

    print("\n[TEST 1] Zero error, default gains")
    x_d = np.array([0.3, 0.2, 0.4])
    x   = x_d.copy()
    w = ci.update(dt, x_d, R_I, x, R_I)
    print("wrench =", w)

    print("\n[TEST 2] Pure position error in +X, no orientation error")
    ci.reset()
    x_d = np.array([0.35, 0.2, 0.4])   # desired 5cm ahead
    x   = np.array([0.30, 0.2, 0.4])   # current
    w = ci.update(dt, x_d, R_I, x, R_I)
    print("e_pos =", x_d - x)
    print("wrench =", w, "(Fx should be ≈ Kx * 0.05)")




    print("\n[TEST 3] Orientation error around Z (10 deg), no position error")
    ci.reset()
    angle = np.deg2rad(10)
    R_d = R.from_euler('z', angle).as_matrix()
    R_m = np.eye(3)  # current = identity, desired = rotated
    x_d = np.zeros(3)
    x   = np.zeros(3)
    w = ci.update(dt, x_d, R_d, x, R_m)
    print("e_ori =", so3_log(R_d.T @ R_m))
    print("wrench =", w, "(Mz should be non-zero)")

    print("\n[TEST 4] Zero gains -> only bias should appear")
    ci.reset()
    ci.set_params(K_pos=(0,0,0), D_pos=(0,0,0), Md_pos=(0,0,0),
                  K_ori=(0,0,0), D_ori=(0,0,0), Md_ori=(0,0,0))
    F_bias = np.array([1.0, 0, 0,  0,0,0])
    w = ci.update(dt,
                  x_d=np.zeros(3), R_d=R_I,
                  x=np.zeros(3),   R_m=R_I,
                  F_bias=F_bias)
    print("wrench =", w, "(should be [1,0,0, 0,0,0])")

    print("\n[INFO] Demo complete.")
