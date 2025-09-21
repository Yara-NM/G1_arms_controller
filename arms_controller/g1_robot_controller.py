"""
G1 Unitree Robot Controller using joint_controller_2layers
"""
import time, os
import numpy as np
from .joint_controller_updated import UnitreeG1ArmController as ctrl_G1_2
from .joint_controller import UnitreeG1ArmController as ctrl_G1_1
from .g1_ik_solver_with_vis import G1_IK_Arms
from .plotting import plot_joint_log , G1_ARM_JOINTS
from .utils import remap_ik_joints_to_motor, map_motor_state
from scipy.spatial.transform import Rotation as R


class G1RobotArmController:
    def __init__(self, 
                 ctrl_dt=0.02 ,
                 ctrl_2l_dt = 0.05, 
                 results_dir = None,  
                 mode = 'h',
                 visualize = False
                 ):
        """
        ctrl_dt: inner-loop timestep for joint controller (s) 
        ctrl_2l_dt: outer-loop trajectory update dt (s) recommended 0.2 sec
        mode: 'h' or 'l' control mode passed to joint controller 
            h for high_level topics 
            l for low_level topics
        jc_results_dir: path for joint controller logs
        """

        self.ctrl_2l = ctrl_2l_dt
        
        #initialize the directory
        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), "results")

        # Initialize IK
        self.ik_solver = G1_IK_Arms(visualize = visualize)

        if ctrl_2l_dt != None:
            self.joint_controller = ctrl_G1_2(control_dt= ctrl_dt,
                                               controller_layers_dt =ctrl_2l_dt ,
                                                results_dir= self.results_dir,
                                                dds_topic= mode)
            
        else: self.joint_controller = ctrl_G1_1(control_dt=ctrl_dt, 
                                                results_dir=self.results_dir,
                                                dds_topic= mode)
        time.sleep(0.5)

        # Store motors state in np.array of shape (nq,)
        self.current_config = None
        self.update_current_config()  
    

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
        
    ####### new function#######

    def IK_join_arms(self, left_tf, right_tf, current_config):
        """
        Compute IK for both arms given 4x4 left_tf and right_tf,
        then update motor targets.
        """
        # get current config if not provided
        # self.update_current_config()  
        q0 = current_config
        # solve IK (returns q_dict, tau_dict)
        q_dict, tau_dict, ok = self.ik_solver.ik_both_pose(left_tf, right_tf, q_init=q0)
        
        return q_dict, tau_dict, ok 
    
    ####### until here the new function #######
    
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
