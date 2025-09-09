import numpy as np
import time, math, csv, os
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

G1_NUM_MOTOR = 30

LOWLEVEL = 0xFF
PosStopF = 2.146e9
VelStopF = 16000.0

# Joint mapping as a dictionary: joint name -> motor index

joint_mapping = {
    # Left Arm
    "LeftShoulderPitch": 15,
    "LeftShoulderRoll": 16,
    "LeftShoulderYaw": 17,
    "LeftElbow": 18,
    "LeftWristRoll": 19,
    "LeftWristPitch": 20,
    "LeftWristYaw": 21,
    # right Arm
    "RightShoulderPitch": 22,
    "RightShoulderRoll": 23,
    "RightShoulderYaw": 24,
    "RightElbow": 25,
    "RightWristRoll": 26,
    "RightWristPitch": 27,
    "RightWristYaw": 28,
    "WaistYaw": 12,
    #weight
    "NotUsedJoint": 29
}

arm_joint_names = [
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWristYaw","NotUsedJoint", "WaistYaw"
]
weak_motors_indices = list(joint_mapping.values())



class UnitreeG1ArmController: 
    def __init__(self, control_dt=0.02, controller_layers_dt =0.2 , results_dir=None, dds_topic="l"): 

        self.control_dt_ = control_dt
        self.controller_layers_dt_ = controller_layers_dt

        self.target_positions = {joint: 0.0 for joint in arm_joint_names}
        self.target_positions["NotUsedJoint"] = 1.0
        self.pending_targets = self.target_positions.copy()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.time_ = 0.0
        self.running = True

        # Per-joint gains for real robot: 
        self.Kp_map = {
        "LeftShoulderPitch": 52.0, "RightShoulderPitch": 52.0,
        "LeftShoulderRoll":  52.0, "RightShoulderRoll":  52.0,
        "LeftShoulderYaw":   26.0, "RightShoulderYaw":   26.0,
        "LeftElbow":         72.8, "RightElbow":         72.8,
        "LeftWristRoll":     26.0, "RightWristRoll":     26.0,
        "LeftWristPitch":    39.0, "RightWristPitch":    39.0,
        "LeftWristYaw":      26.0, "RightWristYaw":      26.0,
        "NotUsedJoint": 35.0, "WaistYaw": 35.0
    }
        self.Kd_map = {
        "LeftShoulderPitch": 1.7, "RightShoulderPitch": 1.7,
        "LeftShoulderRoll":  1.7, "RightShoulderRoll":  1.7,
        "LeftShoulderYaw":   1.2, "RightShoulderYaw":   1.2,
        "LeftElbow":         1.5, "RightElbow":         1.5,
        "LeftWristRoll":     1.2, "RightWristRoll":     1.2,
        "LeftWristPitch":    1.2, "RightWristPitch":    1.2,
        "LeftWristYaw":      1.2, "RightWristYaw":      1.2,
        "NotUsedJoint": 1.0, "WaistYaw": 1.0
    }
    #     # Per-joint gains for simulation: 
        # self.Kp_map = {joint: 30.0 for joint in arm_joint_names}
        # self.Kd_map = {joint: 1.0 for joint in arm_joint_names}
        # for joint in ["LeftWristRoll", "LeftWristPitch", "LeftWristYaw", "RightWristRoll", "RightWristPitch", "RightWristYaw"]:
        #     self.Kp_map[joint] = 20.0
        #     self.Kd_map[joint] = 0.8


        self.tau_ff_map = {j: 0.0 for j in arm_joint_names}
        self.ff_enabled = True
        self.ff_gain = 0.8         # start modest; we’ll tune 0.4–0.8
        self.ff_alpha = 0.9        # low‑pass on τ_ff updates
        # conservative torque clamps (Nm) – adjust if you know motor limits
        self.tau_limit = {
            "LeftShoulderPitch": 25.0, "RightShoulderPitch": 25.0,
            "LeftShoulderRoll":  25.0, "RightShoulderRoll":  25.0,
            "LeftShoulderYaw":   20.0, "RightShoulderYaw":   20.0,
            "LeftElbow":         20.0, "RightElbow":         20.0,
            "LeftWristRoll":     10.0, "RightWristRoll":     10.0,
            "LeftWristPitch":    10.0, "RightWristPitch":    10.0,
            "LeftWristYaw":      10.0, "RightWristYaw":      10.0,         
        }
                


        # Logs
        self.log_data = []
        self.logging_enabled = False

        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.results_dir, f"joint_log_{timestamp}.csv")
        
        # intialize msgs
        self.dds_topic_type = dds_topic
        self._init_low_cmd()
        self._init_topics(dds_topic)
        self.control_thread = None
        self.outer_controller_thread = None

        self._initialize_targets_from_current_state()
        
    def _init_low_cmd(self):
        # Set message header and default values.
        self.low_cmd.level_flag = LOWLEVEL
        self.low_cmd.gpio = 0
        for i in range(G1_NUM_MOTOR):
            
            self.low_cmd.motor_cmd[i].mode = 0x01 if i in weak_motors_indices else 0x0A
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _init_topics(self, topic_type):
        topic_map = {"l": "rt/lowcmd", "h": "rt/arm_sdk"}
        topic_name = topic_map.get(topic_type, "rt/lowcmd")

        self.armcmd_publisher = ChannelPublisher(topic_name, LowCmd_)
        self.armcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        # if topic_type == "l":
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def _initialize_targets_from_current_state(self):
        if self.low_state is None:
            print("[WARN] Cannot initialize targets, low_state not available.")
            return
        for joint in arm_joint_names:
            q = self.low_state.motor_state[joint_mapping[joint]].q
            self.target_positions[joint] = q
            self.pending_targets[joint] = q
        self.pending_targets["NotUsedJoint"] = 1.0
        self.target_positions["NotUsedJoint"] = 1.0
            

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    
    def write_arm_command(self):
        """
        Called periodically by the control loop.
        It updates only the arm joints to move toward their target positions,
        and sets the "weight" parameter in the NotUsedJoint.
        """
        if self.low_state is None:
            return
        
        log_entry = {"time": time.time()}
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.low_state.mode_machine

        # Update arm joints.
        for joint in arm_joint_names:
            idx = joint_mapping[joint]
            current_q = self.low_state.motor_state[idx].q
            target_q = self.target_positions.get(joint, current_q)
            
            self.low_cmd.motor_cmd[idx].q = target_q
            self.low_cmd.motor_cmd[idx].dq = 0.0
            self.low_cmd.motor_cmd[idx].kp = self.Kp_map[joint]
            self.low_cmd.motor_cmd[idx].kd = self.Kd_map[joint]
            if self.ff_enabled:
                lim = self.tau_limit.get(joint, 10.0)
                self.low_cmd.motor_cmd[idx].tau = np.clip(self.ff_gain * self.tau_ff_map[joint],
                                                           -lim, +lim)
            else: self.low_cmd.motor_cmd[idx].tau = 0.0


            # save logs
            if self.logging_enabled:
                log_entry[f"{joint}_target"] = target_q
                log_entry[f"{joint}_pos"] = self.low_state.motor_state[idx].q
                log_entry[f"{joint}_vel"] = self.low_state.motor_state[idx].dq
                log_entry[f"{joint}_tau"] = self.low_state.motor_state[idx].tau_est
                log_entry[f"{joint}_step_target"] = self.target_positions[joint]
                log_entry[f"{joint}_final_target"] = self.pending_targets[joint]

        # save state values
        if self.logging_enabled:
            self.log_data.append(log_entry)
 

        # Update the weight pasrameter using NotUsedJoint.
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = self.target_positions.get("NotUsedJoint", 1.0)

        # Compute and attach the CRC.
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # Write the command over the high-level arm topic.
        self.armcmd_publisher.Write(self.low_cmd)

    def stepwise_update_target_positions(self, step_size_rad=math.radians(3)):
        for joint in arm_joint_names:
            current = self.target_positions[joint]
            desired = self.pending_targets[joint]
            delta = desired - current
            if abs(delta) < step_size_rad:
                self.target_positions[joint] = desired
            else:
                self.target_positions[joint] += step_size_rad * np.sign(delta)

    def start_control_loop(self):
        """
        Start a control loop that continuously sends arm commands.
        """
        self.running = True
        self.control_thread = RecurrentThread(
            interval=self.control_dt_, target=self.write_arm_command, name="g1_arm_control_loop"
        )
        self.control_thread.Start()

        self.outer_controller_thread = RecurrentThread(
            interval=self.controller_layers_dt_, target=self.stepwise_update_target_positions, name="slow_trajectory_updater"
        )
        self.outer_controller_thread.Start()


    def stop_control_loop(self):
        """
        Stop the arm control loop.
        """
        self.running = False
        if self.control_thread is not None:
            self.control_thread.Wait()  # Proper way to request loop exit and join
        # self.release_arm_sdk()
        
        # Stop or join your control thread as appropriate.
    def release_arm_sdk(self):
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.armcmd_publisher.Write(self.low_cmd)

    def read_motor_state(self):
        """
        Returns a dictionary of the current motor positions (radians) for arm joints.
        """
        if self.low_state is None:
            return {}
        state = {}
        for joint in arm_joint_names:
            idx = joint_mapping[joint]
            state[joint] = self.low_state.motor_state[idx].q
        return state
    

    def read_torque_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].tau_est for joint, idx in joint_mapping.items()}
    
    
    def update_target_positions(self, new_targets: dict):
        """ G1 SDK2, 
        Update the target positions for the arm joints.
        Expect keys from the arm joint list or "NotUsedJoint".
        """
        for joint, target in new_targets.items():
            if joint in self.pending_targets:
                self.pending_targets[joint] = target
            else:
                print(f"Warning: {joint} not found in target positions.")

    def update_feedforward_torque(self, tau_map: dict, alpha=None):
        """Low‑pass update of per‑joint feed‑forward torque (Nm)."""
        a = self.ff_alpha if alpha is None else float(alpha)
        for j, v in tau_map.items():
            if j in self.tau_ff_map:
                try:
                    v = float(v)
                except Exception:
                    continue
                self.tau_ff_map[j] = (1.0 - a) * self.tau_ff_map[j] + a * v


    def print_imu_state(self):
        """
        Returns the current IMU sensor values: RPY, gyroscope, and accelerometer.
        return: 
        {
            "rpy": (roll, pitch, yaw),
            "gyroscope": (x, y, z),
            "accelerometer": (x, y, z)
        }
        """
        if self.low_state and hasattr(self.low_state, "imu_state"):
            imu = self.low_state.imu_state
            rpy = imu.rpy
            gyro = imu.gyroscope
            accel = imu.accelerometer
            imu_data = {
                "rpy": tuple(rpy),
                "gyroscope": tuple(gyro),
                "accelerometer": tuple(accel)
            }
            # print(f"[IMU RPY]         Roll: {rpy[0]:.4f}, Pitch: {rpy[1]:.4f}, Yaw: {rpy[2]:.4f}")
            # print(f"[IMU Gyroscope]   X: {gyro[0]:.4f}, Y: {gyro[1]:.4f}, Z: {gyro[2]:.4f}")
            # print(f"[IMU Accelerometer] X: {accel[0]:.4f}, Y: {accel[1]:.4f}, Z: {accel[2]:.4f}")
            return imu_data
        
        else:
            print("IMU state not available yet.")
            return None

    def enable_logging(self, filename = None):
        self.logging_enabled = True
        if filename is not None:
            self.log_filename = filename
        self.log_data = []

    def save_log_to_csv(self):
        if not self.logging_enabled or not self.log_data:
            return
        keys = self.log_data[0].keys()
        with open(self.log_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"[INFO] Log saved to {self.log_filename}")


    def estimate_total_motion_time(self):
        """
        Estimate the time needed for all arm joints to reach their respective pending targets,
        based on the configured step size and outer control loop interval.
        Returns the maximum time required across all joints.
        """
        if self.low_state is None:
            print("[WARN] Low state not yet received.")
            return 0.0

        step_size_rad = math.radians(3)  # Same as used in stepwise_update_target_positions
        max_steps = 0

        for joint in arm_joint_names:
            current = self.low_state.motor_state[joint_mapping[joint]].q
            target = self.pending_targets[joint]
            delta = abs(target - current)
            steps_needed = math.ceil(delta / step_size_rad)
            max_steps = max(max_steps, steps_needed)

        return max_steps * self.controller_layers_dt_
    
    def update_gains(self, gains: dict, which: str):
        """
        Update Kp or Kd for one or more joints.

        Args:
            gains (dict): { "JointName": value, ... }
            which (str):  "kp" or "kd" (case-insensitive)

        Returns:
            dict: {
                "updated": {joint: value, ...},
                "unknown_joints": [ ... ],
                "which": "kp" or "kd"
            }
        """
        if which is None:
            raise ValueError("Specify which='kp' or 'kd'.")
        which = which.lower()
        if which not in ("kp", "kd"):
            raise ValueError("Argument 'which' must be 'kp' or 'kd'.")

        updated = {}
        unknown = []

        for joint, val in gains.items():
            if joint not in self.Kp_map:  # Kp_map and Kd_map share the same keys
                unknown.append(joint)
                continue

            try:
                v = float(val)
            except (TypeError, ValueError):
                print(f"[WARN] Gain for {joint} must be a number. Skipped.")
                continue

            # Optional: light sanity clamp to avoid absurd values
            if not np.isfinite(v):
                print(f"[WARN] Gain for {joint} is not finite. Skipped.")
                continue
            if v < 0.0:
                print(f"[WARN] Negative gain for {joint} is not allowed. Skipped.")
                continue

            if which == "kp":
                self.Kp_map[joint] = v
            else:
                self.Kd_map[joint] = v

            updated[joint] = v

        if unknown:
            print(f"[WARN] Unknown joint names: {unknown}")

        return {"updated": updated, "unknown_joints": unknown, "which": which}

# ===== Helpers (simple, no fancy Python) =====
def wait_for_state(controller, timeout=5.0):
    t0 = time.time()
    while controller.low_state is None and (time.time() - t0) < timeout:
        time.sleep(0.01)
    if controller.low_state is None:
        raise RuntimeError("LowState not received; check NIC/topic/mode.")

def zero_pose(control_joints):
    pose = {}
    for j in control_joints:
        pose[j] = 0.0
    return pose

def move_and_hold(controller, targets, hold_s=0.8):
    controller.update_target_positions(targets)
    t_est = controller.estimate_total_motion_time()
    time.sleep(t_est + hold_s)

def read_and_report(controller, tag, targets):
    state = controller.read_motor_state()
    errs = {}
    for j in targets:
        if j in state:
            errs[j] = abs(targets[j] - state[j])
    if not errs:
        print("[%s] No joint states." % tag)
        return
    mean_err = sum(errs.values()) / float(len(errs))
    max_j = max(errs, key=lambda k: errs[k])
    print("[%s] mean|error| = %.4f rad, max|error| = %.4f rad @ %s" %
          (tag, mean_err, errs[max_j], max_j))

def test_joint_sequence(controller, joint, angles, control_joints, settle=0.8, base_overrides=None):
    """
    base_overrides: dict of joint->rad that will be applied in addition to zero_pose()
                    before commanding 'joint' to each angle.
    """
    print("\n--- Testing %s ---" % joint)
    for a in angles:
        pose = zero_pose(control_joints)
        if base_overrides:
            for jb, vb in base_overrides.items():
                pose[jb] = float(vb)
        pose[joint] = float(a)
        move_and_hold(controller, pose, hold_s=settle)
        read_and_report(controller, "%s -> %.3f" % (joint, a), pose)

def main():
    print("[INFO] DDS init...")
    ChannelFactoryInitialize(0, "enp2s0")   # <- use your real NIC on hardware
    # ChannelFactoryInitialize(1, "lo") 
    time.sleep(0.5)

    print("[INFO] Controller init...")
    controller = UnitreeG1ArmController(control_dt=0.02,
                                        controller_layers_dt=0.1,
                                        dds_topic="h")  # arm SDK path
    controller.start_control_loop()
    wait_for_state(controller)
    controller.enable_logging()

    # Keep these OUT of poses/updates
    LOCKED_JOINTS = ["WaistYaw", "NotUsedJoint"]
    CONTROL_JOINTS = [j for j in arm_joint_names if j not in LOCKED_JOINTS]

    # Safety: lock waist at zero, keep weight channel on
    controller.update_target_positions({"WaistYaw": 0.0})
    controller.target_positions["WaistYaw"] = 0.0
    controller.pending_targets["WaistYaw"] = 0.0
    controller.target_positions["NotUsedJoint"] = 1.0
    controller.pending_targets["NotUsedJoint"] = 1.0

    # === Final gains (from your best sweep) ===
    kp_final = {
        "LeftShoulderPitch": 52.0, "RightShoulderPitch": 52.0,
        "LeftShoulderRoll":  52.0, "RightShoulderRoll":  52.0,
        "LeftShoulderYaw":   26.0, "RightShoulderYaw":   26.0,
        "LeftElbow":         72.8, "RightElbow":         72.8,
        "LeftWristRoll":     26.0, "RightWristRoll":     26.0,
        "LeftWristPitch":    39.0, "RightWristPitch":    39.0,
        "LeftWristYaw":      26.0, "RightWristYaw":      26.0,
    }
    kd_final = {
        "LeftShoulderPitch": 1.7, "RightShoulderPitch": 1.7,
        "LeftShoulderRoll":  1.7, "RightShoulderRoll":  1.7,
        "LeftShoulderYaw":   1.2, "RightShoulderYaw":   1.2,
        "LeftElbow":         1.5, "RightElbow":         1.5,
        "LeftWristRoll":     1.2, "RightWristRoll":     1.2,
        "LeftWristPitch":    1.2, "RightWristPitch":    1.2,
        "LeftWristYaw":      1.2, "RightWristYaw":      1.2,
    }

    # Apply gains once
    print("[INFO] Applying final Kp/Kd maps...")
    controller.update_gains(kp_final, which="kp")
    controller.update_gains(kd_final, which="kd")

    # --- Baseline at ZERO ---
    ZERO = zero_pose(CONTROL_JOINTS)
    print("\n[BASELINE] Moving to ZERO...")
    move_and_hold(controller, ZERO, hold_s=1.0)
    read_and_report(controller, "BASELINE -> ZERO", ZERO)

    # === Per-joint tests (elbows + wrists) ===
    # Choose angles that load joints without hitting limits (adjust if needed)
    elbow_angles = [-0.3, -0.6, -0.9]           # flexing increases load
    wrist_pitch_angles = [-0.5, 0.0, 0.5]
    wrist_roll_angles  = [-0.4, 0.0, 0.4]
    wrist_yaw_angles   = [-0.6, 0.0, 0.6]

    JOINTS_TO_TEST = [
        ("LeftElbow", elbow_angles),
        ("RightElbow", elbow_angles),
        ("LeftWristPitch", wrist_pitch_angles),
        ("RightWristPitch", wrist_pitch_angles),
        ("LeftWristRoll", wrist_roll_angles),
        ("RightWristRoll", wrist_roll_angles),
        ("LeftWristYaw", wrist_yaw_angles),
        ("RightWristYaw", wrist_yaw_angles),
    ]

    for joint, angles in JOINTS_TO_TEST:
        if joint == "LeftWristYaw":
            test_joint_sequence(
                controller, joint, angles, CONTROL_JOINTS, settle=0.8,
                base_overrides={"LeftWristRoll": math.pi/2}
            )
        elif joint == "RightWristYaw":
            test_joint_sequence(
                controller, joint, angles, CONTROL_JOINTS, settle=0.8,
                base_overrides={"RightWristRoll": math.pi/2}
            )
        else:
            test_joint_sequence(controller, joint, angles, CONTROL_JOINTS, settle=0.8)

        # Return to ZERO between joints
        move_and_hold(controller, ZERO, hold_s=0.8)
        read_and_report(controller, "BACK TO ZERO", ZERO)

    # Finish
    controller.stop_control_loop()
    controller.save_log_to_csv()
    print("\n[INFO] Per-joint test complete. CSV saved.")

if __name__ == "__main__":
    main()

# # ==== main section ====

# # Joints we won't command via poses/gain sweeps
# LOCKED_JOINTS = ["WaistYaw", "NotUsedJoint"]
# CONTROL_JOINTS = [j for j in arm_joint_names if j not in LOCKED_JOINTS]

# def zero_pose():
#     # Only arms; excludes waist + weight channel
#     return {j: 0.0 for j in CONTROL_JOINTS}

# def move_and_hold(controller, targets: dict, hold_s=0.8):
#     controller.update_target_positions(targets)
#     t_est = controller.estimate_total_motion_time()
#     time.sleep(t_est + hold_s)

# def read_and_report(controller, name, targets):
#     state = controller.read_motor_state()
#     errs = {}
#     for j, tgt in targets.items():
#         if j in state:
#             errs[j] = float(abs(tgt - state[j]))
#     if not errs:
#         print(f"[{name}] No joint states available"); return
#     mean_err = sum(errs.values()) / len(errs)
#     max_j = max(errs, key=lambda k: errs[k])
#     print(f"[{name}] mean|error| = {mean_err:.4f} rad, max|error| = {errs[max_j]:.4f} rad @ {max_j}")
#     focus = ["LeftElbow", "RightElbow", "LeftWristPitch", "RightWristPitch"]
#     print("          elbows/wrists -> " + "  ".join([f"{j}:{errs.get(j, float('nan')):.4f}" for j in focus]))


# if __name__ == "__main__":

#     print("[INFO] DDS init...")
#     ChannelFactoryInitialize(0,  "enp2s0")    
#     # ChannelFactoryInitialize(1, "lo")
#     time.sleep(0.5)

#     print("[INFO] Controller init...")
#     controller = UnitreeG1ArmController(control_dt=0.02, 
#                                         controller_layers_dt=0.1, 
#                                         dds_topic="h")
#     controller.start_control_loop()
#     time.sleep(1.0)

#     # Right after controller.start_control_loop() and a short sleep:
#     controller.update_target_positions({"WaistYaw": 0.0})
#     controller.target_positions["WaistYaw"] = 0.0
#     controller.pending_targets["WaistYaw"] = 0.0

#     controller.target_positions["NotUsedJoint"] = 1.0
#     controller.pending_targets["NotUsedJoint"] = 1.0

#     # Optional logging
#     controller.enable_logging()

#     # --- Test set: poses and gain configs ---
#     poses = {
#         "ZERO": zero_pose(),
#         "REACH_FWD": {
#             **zero_pose(),
#             "LeftShoulderPitch": 0.6, "LeftElbow": -0.8,
#             "RightShoulderPitch": 0.6, "RightElbow": -0.8,
#         },
#         "LIFT_SIDE": {
#             **zero_pose(),
#             "LeftShoulderRoll": 0.5, "LeftElbow": -0.6,
#             "RightShoulderRoll": -0.5, "RightElbow": -0.6,
#         },
#     }

#     gain_trials = [
#         ("BASE_GAINS", None),  # keep your defaults
#         ("KP_UP_30%", {"which": "kp", "gains": {j: controller.Kp_map[j] * 1.3 for j in CONTROL_JOINTS}}),
#         ("KD_UP_50%", {"which": "kd", "gains": {j: controller.Kd_map[j] * 1.5 for j in CONTROL_JOINTS}}),
#         # Example: stronger elbows/wrists only (heavier links)
#         ("ELBOW_WRIST_BOOST", {"which": "kp", "gains": {
#             "LeftElbow": controller.Kp_map["LeftElbow"] * 1.6,
#             "RightElbow": controller.Kp_map["RightElbow"] * 1.6,
#             "LeftWristPitch": controller.Kp_map["LeftWristPitch"] * 1.5,
#             "RightWristPitch": controller.Kp_map["RightWristPitch"] * 1.5,
#         }}),
#     ]

#     print("\n========== Gain/Accuracy Sweep ==========")
#     # Always return to zero before starting sweeps
#     move_and_hold(controller, poses["ZERO"])
#     read_and_report(controller, "AT_ZERO (pre)", poses["ZERO"])

#     for label, cfg in gain_trials:
#         if cfg is not None:
#             res = controller.update_gains(cfg["gains"], which=cfg["which"])
#             print(f"\n[GAINS] {label}: updated {len(res['updated'])} joints ({res['which']})"
#                   f"{' | unknown: ' + str(res['unknown_joints']) if res['unknown_joints'] else ''}")
#         else:
#             print(f"\n[GAINS] {label}: using default maps")

#         # Test each pose, report accuracy
#         for pose_name, pose in poses.items():
#             move_and_hold(controller, pose)
#             read_and_report(controller, f"{label} -> {pose_name}", pose)

#         # Return to zero after each gain condition
#         move_and_hold(controller, poses["ZERO"])
#         read_and_report(controller, f"{label} -> BACK_TO_ZERO", poses["ZERO"])

#     controller.stop_control_loop()
#     controller.save_log_to_csv()
#     print("\n[INFO] Sweep complete. CSV saved.")




# if __name__ == "__main__":

#     print("[INFO] Initializing DDS...")
#     # ChannelFactoryInitialize(1, "lo")
#     ChannelFactoryInitialize(0, "enp2s0")
#     time.sleep(0.5)

#     print("[INFO] Initializing controller...")
#     controller = UnitreeG1ArmController(control_dt=0.02, controller_layers_dt=0.1, dds_topic="h")
#     controller.start_control_loop()
#     time.sleep(1.0)

#     print("[INFO] Starting realistic joint demo...")
#     controller.enable_logging()

#     # Define a realistic target configuration
#     target_pose = {
#         "LeftShoulderPitch": 0.4,
#         "LeftShoulderRoll": 0.3,
#         "LeftShoulderYaw": 0.2,
#         "LeftElbow": -0.6,
#         "LeftWristRoll": 0.0,
#         "LeftWristPitch": 0.4,
#         "LeftWristYaw": 0.1,
#         "RightShoulderPitch": 0.4,
#         "RightShoulderRoll": -0.3,
#         "RightShoulderYaw": -0.2,
#         "RightElbow": -0.6,
#         "RightWristRoll": 0.0,
#         "RightWristPitch": 0.4,
#         "RightWristYaw": -0.1,
#         "WaistYaw": 0.2
#     }

#     # Move from 0 → target_pose
#     print("[STEP] Moving to target pose...")
#     controller.update_target_positions(target_pose)
#     time.sleep(controller.estimate_total_motion_time() + 1.0)

#     # Move back to zero
#     print("[STEP] Returning to zero pose...")
#     controller.update_target_positions({joint: 0.0 for joint in arm_joint_names})
#     time.sleep(controller.estimate_total_motion_time() + 1.0)

#     # Stop and save
#     controller.stop_control_loop()
#     controller.save_log_to_csv()
#     print("[INFO] Demo complete.")

#     print("\n[INFO] Theoretical Note:")
#     print("- With control_dt = 0.02 s (50 Hz), a 0.5 rad move takes ≈ 0.5/step_size/50 = 114 steps ≈ 2.3 sec")
#     print("- Don't lower control_dt too much (< 0.005 s), may overload CPU or network")
#     print("- Recommended: keep control_dt ≈ 0.01–0.02 s, outer layer ≈ 0.2 s")
    
    

            
