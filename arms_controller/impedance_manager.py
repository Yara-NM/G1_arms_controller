# impedance_manager.py
import numpy as np
from arms_controller.cartesian_impedance import CartesianImpedance

class ImpedanceManager:
    """
    Glue between: joint_controller (I/O), IK (FK+J+tau_g), and task-space impedance.
    Assumes WORLD-frame Jacobians and wrenches.
    """
    def __init__(self, ik, joint_ctrl, utils_module, ff_scale=1.0):
        """
        INPUT:
            ik          : instance of G1_IK_Arms
            joint_ctrl  : instance of UnitreeG1ArmController (one-layer)
            utils_module: your utils.py module (must provide map/remap helpers)
            ff_scale    : scalar to scale the final torque before sending (keep 1.0)
        Owns imp_R, imp_L and default biases off.
        """
        self.ik = ik
        self.joint = joint_ctrl
        self.ut = utils_module
        self.ff_scale = float(ff_scale)

        # Per-arm impedance
        self.imp_R = CartesianImpedance()
        self.imp_L = CartesianImpedance()

        # flags & biases
        self.enable_R = True
        self.enable_L = True
        self.Fbias_R = np.zeros(6)
        self.Fbias_L = np.zeros(6)

        # safety clamps (Nm), override per-joint via joint.set_tau_limits(...)
        self.tau_clip_scalar = None  # optional global scalar cap

    # ---- public setters ------------------------------------------------------
    def enable(self, side, on=True):
        """
        Purpose: Enable/disable impedance per arm (e.g., "R" or "L").
        Inputs: side string, on bool.
        Outputs: None.
        Behavior: If disabled, that arm contributes zero wrench (just gravity remains).
        """
        if side.upper().startswith("R"):
            self.enable_R = bool(on)
        else:
            self.enable_L = bool(on)

    def set_force_bias(self, side, F6):
        """
        Set a constant WORLD-frame wrench bias for that arm, e.g., [0,0, +10, 0,0,0] N to push a button.
        Inputs: side string, F6 array-like shape (6,).
        Outputs: None.
        """
        F6 = np.asarray(F6, float).reshape(6)
        if side.upper().startswith("R"):
            self.Fbias_R = F6
        else:
            self.Fbias_L = F6

    def set_params(self, side, **kw):
        """
        Purpose: Pass through updates for gains/inertias to the chosen arm’s CartesianImpedance.
        Inputs: same keywords as CartesianImpedance.set_params (e.g., K_pos, D_pos, etc.).
        Outputs: None.
        """
        if side.upper().startswith("R"):
            self.imp_R.set_params(**kw)
        else:
            self.imp_L.set_params(**kw)

    def reset_filters(self):
        self.imp_R.reset(); self.imp_L.reset()

    # ---- core step -----------------------------------------------------------
    def step(self, dt,
             xRd, RRd, xLd, RLd,                 # desired poses (world)
             vRd=None, wRd=None, aRd=None, aWRd=None,
             vLd=None, wLd=None, aLd=None, aWLd=None):
        """
        Do one outer control iteration—compute and apply tau_ff into the joint controller.
        One outer-cycle call:
          - read q,dq from joint
          - FK + J
          - impedance per arm (if enabled)
          - tau_ff = J^T F* (R + L) + tau_g
          - clamp + send

        Poses:
          x..: (3,), R..: (3x3) world frame
        Twists default to zero if None.
        """
        # 1) read joint states (motor names) -> IK names
        q_m = self.joint.read_motor_state()
        dq_m = self.joint.read_velocity_state()
        if not q_m:
            return False, "no_state"

        q_fk = self.ut.map_motor_state(q_m)     # motor -> IK joint names
        dq_fk = self.ut.map_motor_state(dq_m)   # same mapping for velocities

        # 2) FK + J
        poses = self.ik.forward_kinematics(q_fk)
        xR = poses["R_ee"].translation; RR = poses["R_ee"].rotation
        xL = poses["L_ee"].translation; RL = poses["L_ee"].rotation

        JR = self.ik.frame_jacobian(q_fk, which="R_ee")  # (6,n)
        JL = self.ik.frame_jacobian(q_fk, which="L_ee")

        # 3) impedance wrenches
        if vRd is None:  vRd  = np.zeros(3)
        if wRd is None:  wRd  = np.zeros(3)
        if aRd is None:  aRd  = np.zeros(3)
        if aWRd is None: aWRd = np.zeros(3)
        if vLd is None:  vLd  = np.zeros(3)
        if wLd is None:  wLd  = np.zeros(3)
        if aLd is None:  aLd  = np.zeros(3)
        if aWLd is None: aWLd = np.zeros(3)

        wR = self.imp_R.update(dt, xRd, RRd, xR, RR, vRd, wRd, aRd, aWRd, self.Fbias_R) if self.enable_R else np.zeros(6)
        wL = self.imp_L.update(dt, xLd, RLd, xL, RL, vLd, wLd, aLd, aWLd, self.Fbias_L) if self.enable_L else np.zeros(6)

        # 4) map to joint torques
        tau_imp_vec = JR.T @ wR + JL.T @ wL  # shape (n,)

        # 5) gravity
        tau_g_vec = self.ik.gravity_torque(q_fk)  # pure gravity (dq=0,ddq=0)

        tau_ff_vec = self.ff_scale * (tau_imp_vec + tau_g_vec)

        # 6) clamp (global scalar; per-joint clamps are in the joint controller)
        if self.tau_clip_scalar is not None:
            lim = float(self.tau_clip_scalar)
            tau_ff_vec = np.clip(tau_ff_vec, -lim, lim)

        # 7) send to joint controller (map vector -> dict -> motor names)
        tau_ff_dict_ik = self.ik._v_to_dict(tau_ff_vec)              # IK joint dict

       # Force/Moment norms already computed 
        tau_R = np.array([v for name, v in tau_ff_dict_ik.items()
                        if name.startswith("right")], dtype=float)

        tau_L = np.array([v for name, v in tau_ff_dict_ik.items()
                        if name.startswith("left")], dtype=float)

        tau_R_norm = float(np.linalg.norm(tau_R)) if tau_R.size > 0 else 0.0
        tau_L_norm = float(np.linalg.norm(tau_L)) if tau_L.size > 0 else 0.0

        tau_ff_motor   = self.ut.remap_ik_joints_to_motor(tau_ff_dict_ik)
        self.joint.enable_ff(True)
        self.joint.update_feedforward_torque(tau_ff_motor)



        return True, {
                        "F_R": float(np.linalg.norm(wR[:3])),
                        "M_R": float(np.linalg.norm(wR[3:])),
                        "F_L": float(np.linalg.norm(wL[:3])),
                        "M_L": float(np.linalg.norm(wL[3:])),
                        "tau_R": tau_R_norm,
                        "tau_L": tau_L_norm,
                    }


if __name__ == "__main__":
    import numpy as np
    from arms_controller.g1_ik_solver_with_vis import G1_IK_Arms
    from arms_controller.joint_controller import UnitreeG1ArmController
    from arms_controller.utils import map_motor_state, remap_ik_joints_to_motor

    class FakeJointController:
        """Minimal fake joint controller for testing impedance manager without robot."""
        def __init__(self):
            # Start in zero pose
            self.q = { "LeftShoulderPitch":0, "LeftShoulderRoll":0, "LeftShoulderYaw":0, 
                       "LeftElbow":0,
                       "RightShoulderPitch":0, "RightShoulderRoll":0, "RightShoulderYaw":0, 
                       "RightElbow":0,
                       "WaistYaw":0 }
            self.dq = {j:0 for j in self.q}
            self.last_tau = None

        def read_motor_state(self):
            return self.q

        def read_velocity_state(self):
            return self.dq

        def enable_ff(self, on):
            pass

        def update_feedforward_torque(self, tau_dict):
            print("[FAKE JC] tau_ff =", tau_dict)
            self.last_tau = tau_dict

    # Initialize fake system
    ik = G1_IK_Arms(visualize=False)
    fake_jc = FakeJointController()

    # Minimal utils stub
    class UT:
        map_motor_state = staticmethod(map_motor_state)
        remap_ik_joints_to_motor = staticmethod(remap_ik_joints_to_motor)

    mgr = ImpedanceManager(ik, fake_jc, UT)

    # Desired pose (slightly forward)
    xR_d = np.array([0.3, -0.2, 0.3])
    RR_d = np.eye(3)
    xL_d = np.array([0.3, +0.2, 0.3])
    RL_d = np.eye(3)

    print("Running test step:")
    ok, info = mgr.step(
        dt=0.01,
        xRd=xR_d, RRd=RR_d,
        xLd=xL_d, RLd=RL_d
    )

    print("Result =", ok, info)
