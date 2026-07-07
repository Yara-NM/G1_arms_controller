import numpy as np
from scipy.spatial.transform import Rotation as R


def so3_log(R_err: np.ndarray) -> np.ndarray:
    """
    Log map from SO(3) to R^3 (axis-angle vector).
    R_err: 3x3 rotation matrix.
    Returns: rotvec (3,), with norm = rotation angle [rad].
    """
    rot = R.from_matrix(R_err)
    ang = rot.magnitude()
    if ang < 1e-12:
        return np.zeros(3)
    return rot.as_rotvec()


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """
    Exp map from R^3 (axis-angle) to SO(3).
    rotvec: (3,) vector, norm = angle [rad].
    Returns: 3x3 rotation matrix.
    """
    return R.from_rotvec(rotvec).as_matrix()


def _to_vec3(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size == 1:
        v = np.repeat(v, 3)
    assert v.size == 3, f"Expected 3 elements, got {v.size}"
    return v


class SpringDamperCartesian:
    """
    Virtual mass–spring–damper in task space that produces a smooth
    desired pose (x_d, R_d) connecting:

        - initial pose:  (x0, R0)   [set via reset()]
        - reference pose: (x_ref, R_ref) [set via set_target()]

    Internal states:
        u_p, u_p_dot    : position displacement & velocity (3,)
        u_r, u_r_dot    : orientation "displacement" & rate in SO(3) log space (3,)

    The actual desired pose is:
        x_d = x0 + u_p
        R_d = exp( u_r ) * R0

    The reference (x_ref, R_ref) is *ramped* internally, so when you change ref
    in the interactive script, the step is smoothed and you avoid jerky jumps.
    """

    def __init__(self,
                 K_p=(15.0, 15.0, 12.0),
                 D_p=(6.0, 6.0, 5.0),
                 M_p=(1.5, 1.5, 1.5),
                 K_r=(3.0, 3.0, 2.0),
                 D_r=(1.0, 1.0, 0.8),
                 M_r=(0.2, 0.2, 0.2),
                 ref_vmax=0.15,
                 ref_omega_max=0.8):
        """
        K_p, D_p, M_p: diagonal mass–spring–damper gains in translation (per axis).
        K_r, D_r, M_r: diagonal mass–spring–damper gains in rotation (per axis, log space).

        ref_vmax      : max speed of reference ramp [m/s].
        ref_omega_max : max angular speed of reference ramp [rad/s].
        """
        self.K_p = _to_vec3(K_p)
        self.D_p = _to_vec3(D_p)
        self.M_p = _to_vec3(M_p)

        self.K_r = _to_vec3(K_r)
        self.D_r = _to_vec3(D_r)
        self.M_r = _to_vec3(M_r)

        self.ref_vmax = float(ref_vmax)
        self.ref_omega_max = float(ref_omega_max)

        # Initial pose
        self.x0 = np.zeros(3)
        self.R0 = np.eye(3)

        # Internal "true" reference (ramped)
        self.x_ref = np.zeros(3)
        self.R_ref = np.eye(3)

        # Commanded reference (instantaneous target from outside)
        self.x_ref_cmd = np.zeros(3)
        self.R_ref_cmd = np.eye(3)

        # Internal states of virtual mass–spring–damper
        self.u_p = np.zeros(3)
        self.u_p_dot = np.zeros(3)

        self.u_r = np.zeros(3)
        self.u_r_dot = np.zeros(3)

    # ------------------------------------------------------------------ #
    # External API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self, x_init: np.ndarray, R_init: np.ndarray):
        """
        Reset internal state so that:
          - initial pose = current measured pose
          - internal reference = initial pose
          - displacements u_p, u_r = 0
        """
        self.x0 = np.asarray(x_init, float).reshape(3)
        self.R0 = np.asarray(R_init, float).reshape(3, 3)

        self.x_ref = self.x0.copy()
        self.R_ref = self.R0.copy()
        self.x_ref_cmd = self.x0.copy()
        self.R_ref_cmd = self.R0.copy()

        self.u_p[:] = 0.0
        self.u_p_dot[:] = 0.0
        self.u_r[:] = 0.0
        self.u_r_dot[:] = 0.0

    def set_target(self, x_ref: np.ndarray, R_ref: np.ndarray):
        """
        Set a NEW desired reference pose. This does NOT instantly jump
        the internal reference. Instead, we store:

            x_ref_cmd, R_ref_cmd

        and ramp (x_ref, R_ref) toward them smoothly inside step().
        """
        self.x_ref_cmd = np.asarray(x_ref, float).reshape(3)
        self.R_ref_cmd = np.asarray(R_ref, float).reshape(3, 3)

    
    # ------------------------------------------------------------------ #
    # Live setters  -- change behavior mid-run without a lurch.   #
    # These mutate only gains / ramp caps; internal state (u_p, u_r) and #
    # the current target are untouched, so the arm keeps its pose and    #
    # only its *behavior* changes. Safe to call from another thread.     #
    # ------------------------------------------------------------------ #
    def set_gains(self, K_p=None, D_p=None, M_p=None,
                  K_r=None, D_r=None, M_r=None):
        """
        Update any subset of the impedance gains in place. Pass None to
        leave a gain unchanged. Each value may be a scalar or a 3-vector.
        """
        if K_p is not None:
            self.K_p = _to_vec3(K_p)
        if D_p is not None:
            self.D_p = _to_vec3(D_p)
        if M_p is not None:
            self.M_p = _to_vec3(M_p)
        if K_r is not None:
            self.K_r = _to_vec3(K_r)
        if D_r is not None:
            self.D_r = _to_vec3(D_r)
        if M_r is not None:
            self.M_r = _to_vec3(M_r)
 
    def set_ref_speed(self, ref_vmax=None, ref_omega_max=None):
        """
        Update the reference-ramp speed caps in place (the *real* speed
        knob in this pipeline). Pass None to leave one unchanged.
        """
        if ref_vmax is not None:
            self.ref_vmax = float(ref_vmax)
        if ref_omega_max is not None:
            self.ref_omega_max = float(ref_omega_max)
 
    def get_config(self):
        """Return a snapshot of current gains + ramp caps (for logging/printing)."""
        return {
            "K_p": self.K_p.copy(), "D_p": self.D_p.copy(), "M_p": self.M_p.copy(),
            "K_r": self.K_r.copy(), "D_r": self.D_r.copy(), "M_r": self.M_r.copy(),
            "ref_vmax": self.ref_vmax, "ref_omega_max": self.ref_omega_max,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _ramp_reference(self, dt: float):
        """
        Move (self.x_ref, self.R_ref) toward (self.x_ref_cmd, self.R_ref_cmd)
        at bounded linear and angular speed.
        """
        # --- position ramp ---
        dx_cmd = self.x_ref_cmd - self.x_ref
        dist = np.linalg.norm(dx_cmd)
        max_step = self.ref_vmax * max(dt, 1e-4)
        if dist > max_step > 0.0:
            self.x_ref += dx_cmd * (max_step / dist)
        else:
            self.x_ref = self.x_ref_cmd.copy()

        # --- orientation ramp ---
        R_err = self.R_ref_cmd @ self.R_ref.T  # from current ref to commanded ref
        rotvec = so3_log(R_err)
        ang = np.linalg.norm(rotvec)
        if ang < 1e-6:
            # Already aligned
            self.R_ref = self.R_ref_cmd.copy()
        else:
            max_dtheta = self.ref_omega_max * max(dt, 1e-4)
            dtheta = min(ang, max_dtheta)
            axis = rotvec / ang
            dR = so3_exp(axis * dtheta)
            # Apply incremental rotation
            self.R_ref = dR @ self.R_ref

    # ------------------------------------------------------------------ #
    # Main step                                                          #
    # ------------------------------------------------------------------ #
    def step(self, dt: float):
        """
        One integration step of the virtual mass–spring–damper + reference ramp.

        Inputs:
            dt : timestep [s]

        Returns:
            x_d : (3,) desired position
            R_d : (3,3) desired orientation
            dbg : dict with internal debug info
        """
        dt = float(dt)
        dt = max(dt, 1e-4)

        # 1) Smoothly move internal reference toward commanded reference
        self._ramp_reference(dt)

        # 2) Position dynamics: u_p tracks (x_ref - x0)
        #    u_p_ddot = (K_p/M_p)*( (x_ref - x0) - u_p ) - (D_p/M_p)*u_p_dot
        x_err_ref = (self.x_ref - self.x0) - self.u_p
        u_p_ddot = (self.K_p / self.M_p) * x_err_ref - (self.D_p / self.M_p) * self.u_p_dot

        self.u_p_dot += u_p_ddot * dt
        self.u_p += self.u_p_dot * dt

        x_d = self.x0 + self.u_p

        # 3) Orientation dynamics in log space:
        #    Let r_ref be the log of R_ref * R0^T
        R_ref_rel = self.R_ref @ self.R0.T
        r_ref = so3_log(R_ref_rel)              # "target" displacement in log space
        r_err = r_ref - self.u_r                # error between target and current
        u_r_ddot = (self.K_r / self.M_r) * r_err - (self.D_r / self.M_r) * self.u_r_dot

        self.u_r_dot += u_r_ddot * dt
        self.u_r += self.u_r_dot * dt

        R_d = so3_exp(self.u_r) @ self.R0

        # 4) Debug info (useful for tuning and plotting)
        dbg = {
            "x_ref_cmd": self.x_ref_cmd.copy(),
            "R_ref_cmd": self.R_ref_cmd.copy(),
            "x_ref": self.x_ref.copy(),
            "u_p": self.u_p.copy(),
            "u_p_dot": self.u_p_dot.copy(),
            "u_r": self.u_r.copy(),
            "u_r_dot": self.u_r_dot.copy(),
        }

        return x_d, R_d, dbg


# ---------------------------------------------------------------------- #
# Small standalone demo (no robot)                                       #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Demo: start at x0 = [0.42671, -0.15184, 0.06149],
          target = x_ref = [0.32686, -0.15184, 0.06149],
          R0 = R_ref = Identity (no rotation change).

    Prints x_d over 5 seconds so you can see the smooth approach.
    """
    import time as _time

    x0 = np.array([0.42671, -0.15184, 0.06149])
    R0 = np.eye(3)
    x_ref = np.array([0.32686, -0.15184, 0.06149])
    R_ref = np.eye(3)

    spring = SpringDamperCartesian(
        K_p=(15.0, 15.0, 12.0),
        D_p=(6.0, 6.0, 5.0),
        M_p=(1.5, 1.5, 1.5),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2),
        ref_vmax=0.15,
        ref_omega_max=0.8,
    )

    spring.reset(x0, R0)
    spring.set_target(x_ref, R_ref)

    print("[DEMO] SpringDamperCartesian with reference ramping.\n")
    t0 = _time.time()
    last_print = 0.0
    dt = 0.02

    for k in range(250):  # ~5 s
        t = _time.time() - t0
        x_d, R_d, dbg = spring.step(dt)

        if t - last_print >= 0.5:
            last_print = t
            print(f"t = {t:4.2f} s | x_d = {x_d}")

        _time.sleep(dt)

    print("\n[DEMO] Done.")
