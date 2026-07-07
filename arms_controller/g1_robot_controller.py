"""

"""
import time, os
import numpy as np
from .joint_controller_updated import UnitreeG1ArmController as ctrl_G1_2
from .joint_controller import UnitreeG1ArmController as ctrl_G1_1
from .g1_ik_solver_with_vis import G1_IK_Arms
from .plotting import plot_joint_log , G1_ARM_JOINTS
from .utils import remap_ik_joints_to_motor, map_motor_state
from arms_controller.impedance_manager import ImpedanceManager
from arms_controller import utils
from scipy.spatial.transform import Rotation as R


class G1RobotArmController:
    def __init__(self, 
                 ctrl_dt=0.02 ,
                 ctrl_2l_dt = 0.05, 
                 results_dir = None,  
                 mode = 'h',
                 visualize = False,
                 imp=False 
                 ):
        """
        ctrl_dt: inner-loop timestep for joint controller (s) 
        ctrl_2l_dt: outer-loop trajectory update dt (s) recommended 0.2 sec
                    Ignored if imp=True.
        mode: 'h' or 'l' control mode passed to joint controller 
            h for high_level topics 
            l for low_level topics
        jc_results_dir: path for joint controller logs
        imp: if True, force one-layer joint controller and set up impedance manager.
        """

        self.ctrl_2l = ctrl_2l_dt
        self.control_dt = ctrl_dt

        #initialize the directory
        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), "results")

        # Initialize IK
        self.ik_solver = G1_IK_Arms(visualize = visualize)

        # --- Decide which joint controller to build ---
        if imp:
            # User explicitly wants impedance-capable behavior.
            # Force the one-layer controller.
            self.joint_controller = ctrl_G1_1(
                control_dt=ctrl_dt,
                results_dir=self.results_dir,
                dds_topic=mode
            )
            self.ctrl_2l = None   # make it clear downstream we are NOT using the 2-layer
            impedance_enabled = True
        else: 
            # Legacy behavior: pick controller based on ctrl_2l_dt
            if ctrl_2l_dt != None:
                # two-layer motion generator
                self.joint_controller = ctrl_G1_2(control_dt= ctrl_dt,
                                                controller_layers_dt =ctrl_2l_dt ,
                                                    results_dir= self.results_dir,
                                                    dds_topic= mode)
                impedance_enabled = False
            else: 
                # single-layer PD+ff controller
                self.joint_controller = ctrl_G1_1(control_dt=ctrl_dt, 
                                                    results_dir=self.results_dir,
                                                    dds_topic= mode)
                impedance_enabled = False

        time.sleep(0.5)

        # Store motors state in np.array of shape (nq,)
        self.current_config = None
        self.update_current_config()  
    
        # --- Mode / impedance manager wiring ---
        # public mode: "position" or "impedance"
        self.mode = "impedance" if impedance_enabled else "position"

        # impedance manager handle
        if impedance_enabled:
            self.imp_mgr = ImpedanceManager(self.ik_solver,
                                            self.joint_controller,
                                            utils,
                                            ff_scale=1.0)
            # fresh filters before we start
            self.imp_mgr.reset_filters()
        else:
            self.imp_mgr = None


    def reset_arms(self):
        neutral_positions = {
            
            "LeftShoulderPitch": 0.0,
            "LeftShoulderRoll": 0.0, 
            "LeftShoulderYaw":0.0, 
            "LeftElbow":0.0,
            "LeftWristRoll":0.0, 
            "LeftWristPitch":0.0, 
            "LeftWristYaw":0.0,

            "RightShoulderPitch":0.0, 
            "RightShoulderRoll":0.0, 
            "RightShoulderYaw":0.0, 
            "RightElbow":0.0,
            "RightWristRoll":0.0, 
            "RightWristPitch":0.0, 
            "RightWristYaw":0.0
        }
        self.joint_controller.update_target_positions(neutral_positions)

    def get_R_ee_pose(self):
        motor_state = self.joint_controller.read_motor_state()
        right_pose = self.ik_solver.forward_kinematics(map_motor_state(motor_state))
        RP = right_pose["R_ee"].translation.copy()
        RR = right_pose["R_ee"].rotation.copy()  
        return RP, RR  
    
    def get_L_ee_pose(self):
        motor_state = self.joint_controller.read_motor_state()
        left_pose = self.ik_solver.forward_kinematics(map_motor_state(motor_state))
        LP = left_pose["L_ee"].translation.copy()
        LR = left_pose["L_ee"].rotation.copy()  
        return LP, LR  
    
    def imu_state (self):
        return self.joint_controller.print_imu_state()

    def stop(self):
        self.joint_controller.stop_control_loop()

    def start (self):
        self.joint_controller.start_control_loop()

    def update_joints (self, joints_dict):
        self.joint_controller.update_target_positions (joints_dict)

    def read_joints (self):
        return self.joint_controller.read_motor_state()
    
    def read_torque(self):
        return self.joint_controller.read_torque_state()
    
    # Unified move function using ik_both_pose
    def move_arms(self, left_tf, right_tf):
        """
        Compute IK for both arms given 4x4 left_tf and right_tf,
        then update motor targets.
        """
        if self.mode == "impedance":
            print("[WARN] move_arms() called while in impedance mode. Ignoring.")
            return False
        # get current config if not provided
        self.update_current_config()  
        q0 = self.current_config
        # solve IK (returns q_dict, tau_dict)
        q_dict, tau_dict, ok = self.ik_solver.ik_both_pose(left_tf, right_tf, q_init=q0)
        tau_ff_motor = remap_ik_joints_to_motor(tau_dict)   # reuse same mapping

        # remap and send commands
        motor_cmd = remap_ik_joints_to_motor(q_dict)
        tau_ff_motor = remap_ik_joints_to_motor(tau_dict)   # reuse same mapping
        # If IK failed, don't push a new (possibly stale) posture; just refresh τ_ff
        if not ok:
            print("[WARN] IK failed; applying gravity τ at current configuration and keeping last targets.")
            self.joint_controller.update_feedforward_torque(tau_ff_motor)   # or set_targets_and_tau with same targets
            return False
        
        else:
            self.joint_controller.update_target_positions(motor_cmd)
            self.joint_controller.update_feedforward_torque(tau_ff_motor)
            return True
        

    # Convenience wrappers for different rotation formats
    def move_arms_with_Rt(self, left_R, left_t, right_R, right_t, q_init=None):
        left_tf = np.eye(4)
        left_tf[:3,:3] = left_R; left_tf[:3,3] = left_t
        right_tf = np.eye(4)
        right_tf[:3,:3] = right_R; right_tf[:3,3] = right_t
        return self.move_arms(left_tf, right_tf)

    def move_arms_with_quat(self, left_quat, left_t, right_quat, right_t, q_init=None):
        left_tf = np.eye(4)
        left_tf[:3,:3] = R.from_quat(left_quat).as_matrix(); left_tf[:3,3] = left_t
        right_tf = np.eye(4)
        right_tf[:3,:3] = R.from_quat(right_quat).as_matrix(); right_tf[:3,3] = right_t
        return self.move_arms(left_tf, right_tf)

    def move_arms_with_rpy(self, left_rpy, left_t, right_rpy, right_t, q_init=None):
        left_tf = np.eye(4)
        left_tf[:3,:3] = R.from_euler('xyz', left_rpy).as_matrix(); left_tf[:3,3] = left_t
        right_tf = np.eye(4)
        right_tf[:3,:3] = R.from_euler('xyz', right_rpy).as_matrix(); right_tf[:3,3] = right_t
        return self.move_arms(left_tf, right_tf)

        # Single-arm moves: preserve the other arm's current pose
    def move_right(self, right_tf):
        """Move only the right arm to right_tf; left arm holds current pose"""
        # get current left end-effector pose
        L_pos, L_rot = self.get_L_ee_pose()
        left_tf = np.eye(4)
        left_tf[:3,:3] = R.from_quat(L_rot).as_matrix() if L_rot.shape == (4,) else L_rot
        left_tf[:3,3] = L_pos
        # call both-arm solver
        return self.move_arms(left_tf, right_tf)

    def move_left(self, left_tf):
        """Move only the left arm to left_tf; right arm holds current pose"""
        # get current right end-effector pose
        R_pos, R_rot = self.get_R_ee_pose()
        right_tf = np.eye(4)
        right_tf[:3,:3] = R.from_quat(R_rot).as_matrix() if R_rot.shape == (4,) else R_rot
        right_tf[:3,3] = R_pos
        # call both-arm solver
        return self.move_arms(left_tf, right_tf)

    
    def update_current_config(self):
        motor_state = self.joint_controller.read_motor_state()
        joint_dict = map_motor_state(motor_state)
        self.current_config = self.ik_solver.joint_dict_to_q(joint_dict)

    def start_logging(self):
        self.joint_controller.enable_logging()

    def save_log(self):
        self.joint_controller.save_log_to_csv()

    
    def plot(self, joint_names=None):
        """
        Plot the latest logged joint data.

        Parameters:
        - joint_names: list of joint names to plot (default: common arm joints)
        """
        if not hasattr(self.joint_controller, "log_filename"):
            print("[WARN] No log filename available in controller.")
            return

        log_file = self.joint_controller.log_filename
        if not os.path.exists(log_file):
            print(f"[WARN] Log file not found: {log_file}")
            return

        if joint_names is None:
            joint_names = G1_ARM_JOINTS
        plot_joint_log(log_file, joint_names, results_dir= self.results_dir)

    def estimate_total_motion_time(self):
        if self.ctrl_2l != None:
            return self.joint_controller.estimate_total_motion_time()
        else: return 2


        # ---- Speed envelope controls ----
    def set_speed_preset(self, name: str):
        """Change motion envelope (v/a/j/λ) atomically."""
        self.joint_controller.set_speed_preset(name)

    def set_speed_scale(self, lam: float):
        """Global λ multiplier (keeps preset shape, scales it)."""
        self.joint_controller.set_global_speed_scale(lam)

    def set_joint_speed_caps(self, caps: dict):
        """
        Per-joint overrides:
        caps = {"LeftElbow": {"v": 30.0, "a": 150.0, "j": 1200.0}, ...}  # deg units
        """
        self.joint_controller.set_joint_speed_caps(caps)

    # ---- Gain controls ----
    def set_auto_damping(self, enabled: bool):
        """If True, Kd tracks Kp via kd_per_sqrt_kp_map; if False, you set Kd explicitly."""
        self.joint_controller.auto_damping = bool(enabled)

    def set_gain_slew(self, kp_slew: float = None, kd_slew: float = None):
        """Change how fast Kp/Kd ramp toward targets (units per second)."""
        if kp_slew is not None:
            self.joint_controller.kp_slew_rate = float(kp_slew)
        if kd_slew is not None:
            self.joint_controller.kd_slew_rate = float(kd_slew)

    def update_kp(self, kp_map: dict):
        """kp_map = {'LeftElbow': 35.0, 'RightShoulderPitch': 52.0, ...}"""
        self.joint_controller.update_gains(kp_map, which="kp")

    def update_kd(self, kd_map: dict):
        """kd_map = {'LeftElbow': 1.6, 'RightShoulderPitch': 1.8, ...}"""
        self.joint_controller.update_gains(kd_map, which="kd")


    def apply_task_profile(self, *, preset: str, kp: dict = None, kd: dict = None,
                           lam: float = None, auto_damping: bool = None,
                           kp_slew: float = None, kd_slew: float = None):
        """
        Atomically apply a motion preset + optional λ and gain targets.
        This is safe to call mid-motion.
        """
        if auto_damping is not None:
            self.set_auto_damping(auto_damping)
        if kp_slew is not None or kd_slew is not None:
            self.set_gain_slew(kp_slew, kd_slew)

        # Speed envelope first (uses _speed_lock inside)
        self.set_speed_preset(preset)
        if lam is not None:
            self.set_speed_scale(lam)

        # Gains next (gain thread will ramp smoothly)
        if kp: self.update_kp(kp)
        if kd: self.update_kd(kd)

    # Impedance Functions 
    def enable_impedance(self, right: bool = True, left: bool = True):
        if self.imp_mgr is None:
            print("[WARN] enable_impedance() called but imp_mgr is None.")
            return
        self.imp_mgr.enable("R", right)
        self.imp_mgr.enable("L", left)

    def set_impedance_params(self, side: str, *, K_pos=None, D_pos=None, Md_pos=None,
                             K_ori=None, D_ori=None, Md_ori=None):
        if self.imp_mgr is None:
            print("[WARN] set_impedance_params() called but imp_mgr is None.")
            return
        self.imp_mgr.set_params(side, K_pos=K_pos, D_pos=D_pos, Md_pos=Md_pos,
                                       K_ori=K_ori, D_ori=D_ori, Md_ori=Md_ori)

    def set_force_bias(self, side: str, F6):
        if self.imp_mgr is None:
            print("[WARN] set_force_bias() called but imp_mgr is None.")
            return
        self.imp_mgr.set_force_bias(side, F6)

    def step_impedance(self, dt, xRd, RRd, xLd, RLd,
                    vRd=None, wRd=None, aRd=None, aWRd=None,
                    vLd=None, wLd=None, aLd=None, aWLd=None):
        """
        One impedance control step:
        (1) Compute + apply tau_ff using impedance laws.
        (2) Solve IK to get q_des and send those joint targets.
        Must be in mode == 'impedance' and initialized with imp=True.
        """
        if self.imp_mgr is None:
            return {"ok": False, "why": "impedance not initialized (imp=False in constructor)"}
        if self.mode != "impedance":
            return {"ok": False, "why": "controller mode is not 'impedance'"}

        # 1) impedance tau_ff pushdown
        ok, info = self.imp_mgr.step(self.control_dt,
                                    xRd, RRd, xLd, RLd,
                                    vRd, wRd, aRd, aWRd,
                                    vLd, wLd, aLd, aWLd)
        if not ok:
            return {"ok": False, "why": info}

        # 2) IK to compute q_des for both arms
        left_tf = np.eye(4);  left_tf[:3,:3] = RLd; left_tf[:3,3] = xLd
        right_tf = np.eye(4); right_tf[:3,:3] = RRd; right_tf[:3,3] = xRd

        # Seed IK with current config
        self.update_current_config()
        q_dict, _tau_g_dict, okIK = self.ik_solver.ik_both_pose(left_tf, right_tf, q_init=self.current_config)

        motor_cmd = remap_ik_joints_to_motor(q_dict)
        self.joint_controller.update_target_positions(motor_cmd)

        out = dict(info)
        out.update({"ok": True, "ik_ok": bool(okIK)})
        return out

