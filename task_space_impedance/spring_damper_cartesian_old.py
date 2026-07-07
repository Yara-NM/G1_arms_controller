# spring_damper_cartesian.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def _to_vec(v, n):
    """
    Normalize scalar/list/array into a 1D numpy vector of length n.

    - v: scalar, list/tuple, or np.ndarray
    - n: desired length (here we always use n=3 for x,y,z or roll,pitch,yaw-like axes)

    If v is a scalar, it is broadcast to n elements.
    """
    v = np.array(v, dtype=float).reshape(-1)
    if v.size == 1:
        v = np.repeat(v, n)
    assert v.size == n, f"Expected {n} elements, got {v.size}"
    return v

def so3_log(R_err):
    """
    Log map for SO(3).

    Input:
        R_err: 3x3 rotation matrix (np.ndarray)
    Output:
        rotvec: np.ndarray shape (3,) containing axis * angle (radians)
    """
    rot = R.from_matrix(R_err)
    ang = rot.magnitude()
    if ang < 1e-12:
        return np.zeros(3)
    return rot.as_rotvec()

def so3_exp(rotvec):
    """
    Exp map for SO(3).

    Input:
        rotvec: np.ndarray shape (3,) axis * angle
    Output:
        R: 3x3 rotation matrix
    """
    return R.from_rotvec(rotvec).as_matrix()

class SpringDamperCartesian:
    """
    Virtual mass–spring–damper system in Cartesian space.

    We maintain internal states (per arm):

        δx      ∈ R^3   position offset from reference
        δx_dot  ∈ R^3   velocity of the offset

        δφ      ∈ R^3   orientation offset (axis-angle) from reference
        δφ_dot  ∈ R^3   rate of orientation offset

    Dynamics (per axis i):

        M_pos[i] * δx_ddot[i] + D_pos[i] * δx_dot[i] + K_pos[i] * δx[i] = F_ext_pos[i]
        M_ori[i] * δφ_ddot[i] + D_ori[i] * δφ_dot[i] + K_ori[i] * δφ[i] = tau_ext_ori[i]

    Then commanded EE pose is:

        x_d = x_ref + δx
        R_d = R_ref * Exp(δφ^)

    where Exp() is the SO(3) exponential map.
    """

    def __init__(self,
                 x_ref=None, R_ref=None,
                 K_p=(20.0, 20.0, 20.0),
                 D_p=(5.0, 5.0, 5.0),
                 M_p=(1.0, 1.0, 1.0),
                 K_r=(4.0, 4.0, 4.0),
                 D_r=(1.0, 1.0, 1.0),
                 M_r=(0.2, 0.2, 0.2)):
        """
        Args:
            x_ref: (3,) reference position [m]. If None, will be set in reset().
            R_ref: (3,3) reference rotation. If None, will be set in reset().
            K_pos, D_pos, M_pos: translational stiffness/damping/mass (3,) or scalar.
            K_ori, D_ori, M_ori: rotational stiffness/damping/mass (3,) or scalar.
        """
        self.K_pos = _to_vec(K_p, 3)
        self.D_pos = _to_vec(D_p, 3)
        self.M_pos = _to_vec(M_p, 3)

        self.K_ori = _to_vec(K_r, 3)
        self.D_ori = _to_vec(D_r, 3)
        self.M_ori = _to_vec(M_r, 3)

        # Reference pose (task-defined target)
        if x_ref is None:
            self.x_ref = np.zeros(3)
        else:
            self.x_ref = np.array(x_ref, dtype=float).reshape(3)

        if R_ref is None:
            self.R_ref = np.eye(3)
        else:
            self.R_ref = np.array(R_ref, dtype=float).reshape(3, 3)

        # Internal states (offsets from reference)
        self.dx = np.zeros(3)       # δx
        self.dx_dot = np.zeros(3)   # δx_dot

        self.dphi = np.zeros(3)     # δφ (axis-angle)
        self.dphi_dot = np.zeros(3) # δφ_dot

    # ------------------------------------------------------------------
    # Configuration / reset
    # ------------------------------------------------------------------
    def set_target(self, x_ref, R_ref):
        """
        Set a new reference pose (task target) without changing current offsets.
        If you want to re-initialize offsets relative to a new reference, call
        reset(...) afterwards.
        """
 
        self.x_ref = np.array(x_ref, dtype=float).reshape(3)
        self.R_ref = np.array(R_ref, dtype=float).reshape(3, 3)

    def reset(self, x_init, R_init):
        """
        Initialize the internal offsets so that:
            x_init = x_ref + δx
            R_init = R_ref * Exp(δφ^)

        That is, we start exactly at the current physical EE pose.

        Args:
            x_init: (3,) initial EE position (e.g. measured)
            R_init: (3,3) initial EE orientation (e.g. measured)
        """
        x_init = np.array(x_init, dtype=float).reshape(3)
        R_init = np.array(R_init, dtype=float).reshape(3, 3)

        # Position offset
        self.dx = x_init - self.x_ref
        self.dx_dot[:] = 0.0

        # Orientation offset: R_ref^T * R_init
        R_err = self.R_ref.T @ R_init
        self.dphi = so3_log(R_err)
        self.dphi_dot[:] = 0.0

    def set_gains(self, K_pos=None, D_pos=None, M_pos=None,
                  K_ori=None, D_ori=None, M_ori=None):
        """
        Update gains/inertias.
        """
        if K_pos is not None: self.K_pos = _to_vec(K_pos, 3)
        if D_pos is not None: self.D_pos = _to_vec(D_pos, 3)
        if M_pos is not None: self.M_pos = _to_vec(M_pos, 3)

        if K_ori is not None: self.K_ori = _to_vec(K_ori, 3)
        if D_ori is not None: self.D_ori = _to_vec(D_ori, 3)
        if M_ori is not None: self.M_ori = _to_vec(M_ori, 3)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(self, dt,
             F_ext_pos=None,
             tau_ext_ori=None):
        """
        Advance the virtual mass–spring–damper by one time step.

        Args:
            dt: float, time step [s].
            F_ext_pos: (3,) external translational force in "virtual space" [N].
                       If None, treated as zero.
            tau_ext_ori: (3,) external rotational torque in "virtual space" [Nm].
                         If None, treated as zero.

        Returns:
            x_d: (3,) commanded EE position [m].
            R_d: (3,3) commanded EE orientation.
            debug: dict with states/accelerations, for logging if needed.
        """
        dt = float(dt)
        if F_ext_pos is None:
            F_ext_pos = np.zeros(3)
        else:
            F_ext_pos = np.array(F_ext_pos, dtype=float).reshape(3)

        if tau_ext_ori is None:
            tau_ext_ori = np.zeros(3)
        else:
            tau_ext_ori = np.array(tau_ext_ori, dtype=float).reshape(3)

        # --- 1. Translational dynamics: M δx¨ + D δx˙ + K δx = F_ext ---
        #    => δx¨ = (F_ext - D δx˙ - K δx) / M  (elementwise)
        ddx = (F_ext_pos - self.D_pos * self.dx_dot - self.K_pos * self.dx) / self.M_pos

        # Semi-implicit Euler integration:
        self.dx_dot += ddx * dt
        self.dx     += self.dx_dot * dt

        # --- 2. Rotational dynamics (in rotvec space): ---
        #    M_ori δφ¨ + D_ori δφ˙ + K_ori δφ = τ_ext
        ddphi = (tau_ext_ori - self.D_ori * self.dphi_dot - self.K_ori * self.dphi) / self.M_ori

        self.dphi_dot += ddphi * dt
        self.dphi     += self.dphi_dot * dt

        # --- 3. Construct commanded pose relative to reference ---
        x_d = self.x_ref + self.dx
        R_d = self.R_ref @ so3_exp(self.dphi)

        debug = {
            "dx": self.dx.copy(),
            "dx_dot": self.dx_dot.copy(),
            "ddx": ddx.copy(),
            "dphi": self.dphi.copy(),
            "dphi_dot": self.dphi_dot.copy(),
            "ddphi": ddphi.copy()
        }

        return x_d, R_d, debug


# ----------------------------------------------------------------------
# Standalone demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Simple demo: start from some initial pose, move to a reference pose
    using the virtual spring–damper, and print the trajectory.

    This does NOT talk to a real robot. It just shows how x_d, R_d evolve.
    Adapt this logic in your task execution script and then feed x_d, R_d
    into your IK + joint controller.
    """
    import time

    # Reference pose (for example, your "zero pose")
    x_ref = np.array([0.32686, -0.15184, 0.06149])
    R_ref = np.array([[ 0.99366,  0.01195,  0.11179],
                      [-0.01149,  0.99992, -0.00482],
                      [-0.11184,  0.00350,  0.99372]])

    # Initial pose is displaced by 10 cm in x and yaw 15 deg
    x_init = x_ref + np.array([0.10, 0.0, 0.0])
    R_init = R.from_euler("z", 15.0, degrees=True).as_matrix() @ R_ref

    # Create spring–damper with gentle gains
    sd = SpringDamperCartesian(
        x_ref=x_ref,
        R_ref=R_ref,
        K_p=(15.0, 15.0, 10.0),
        D_p=(5.0, 5.0, 4.0),
        M_p=(1.0, 1.0, 1.0),
        K_r=(3.0, 3.0, 2.0),
        D_r=(1.0, 1.0, 0.8),
        M_r=(0.2, 0.2, 0.2)
    )

    # Initialize offsets based on initial (measured) pose
    sd.reset(x_init, R_init)

    dt = 0.01  # 100 Hz virtual update
    T  = 5.0   # simulate 5 seconds
    steps = int(T / dt)

    print("[DEMO] SpringDamperCartesian going from x_init to x_ref.\n")
    for k in range(steps):
        x_d, R_d, dbg = sd.step(dt)  # no external forces/torques
        if k % 50 == 0:  # print every 0.5s
            print(f"t = {k*dt:4.2f} s | x_d = {x_d}")
        time.sleep(0.0)  # no real-time constraint here

    print("\n[DEMO] Done.")
