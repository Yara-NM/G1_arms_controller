import numpy as np
import time, math, csv, os
from datetime import datetime
import threading
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
    "RightWristRoll", "RightWristPitch", "RightWristYaw", "WaistYaw", "NotUsedJoint"
]
weak_motors_indices = list(joint_mapping.values())



class UnitreeG1ArmController: 
    def __init__(self, control_dt=0.02, results_dir=None, dds_topic="l"): 

        self.control_dt_ = control_dt
        self.target_positions = {joint: 0.0 for joint in arm_joint_names}
        self.target_positions["NotUsedJoint"] = 1.0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.time_ = 0.0
        self.running = True

    #     # Per-joint gains for real robot: 
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
        # Per-joint gains for Simulation. 
        # self.Kp_map = {joint: 35.0 for joint in arm_joint_names}
        # self.Kd_map = {joint: 1.0 for joint in arm_joint_names}
        # for joint in ["LeftWristRoll", "LeftWristPitch", "LeftWristYaw", "RightWristRoll", "RightWristPitch", "RightWristYaw"]:
        #     self.Kp_map[joint] = 20.0
        #     self.Kd_map[joint] = 0.8
    
        

        # --- Gain targets & smoothing (independent thread) ---
        self.Kp_target_map = dict(self.Kp_map)
        self.Kd_target_map = dict(self.Kd_map)

        # Slew rates (gain units per second)
        self.kp_slew_rate = 20.0     # sim-friendly; lower on real robot (e.g., 60)
        self.kd_slew_rate = 1.0

        self.auto_damping = True
        self.kd_per_sqrt_kp_map = {
                j: (self.Kd_map[j] / math.sqrt(max(self.Kp_map[j], 1e-6)))
                for j in self.Kp_map.keys()
            }

        # Gain update period for the background thread
        self.gain_dt_ = 0.01          # 100 Hz is very smooth

        # Threading
        self._gain_lock = threading.Lock()
        self.gain_thread = None

        # torques: 
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

        # Kp_Kd limits: 

        # Kp caps (≈ 1.5x current)
        self.Kp_map_limit = {
            "LeftShoulderPitch": 78.0, "RightShoulderPitch": 78.0,
            "LeftShoulderRoll":  78.0, "RightShoulderRoll":  78.0,
            "LeftShoulderYaw":   39.0, "RightShoulderYaw":   39.0,
            "LeftElbow":        110.0, "RightElbow":        110.0,
            "LeftWristRoll":     40.0, "RightWristRoll":     40.0,
            "LeftWristPitch":    60.0, "RightWristPitch":    60.0,
            "LeftWristYaw":      40.0, "RightWristYaw":      40.0,
            "WaistYaw":          55.0, "NotUsedJoint":       55.0,
        }

        # Kd caps (≈ 1.5x current)
        self.Kd_map_limit = {
            "LeftShoulderPitch": 2.6, "RightShoulderPitch": 2.6,
            "LeftShoulderRoll":  2.6, "RightShoulderRoll":  2.6,
            "LeftShoulderYaw":   1.8, "RightShoulderYaw":   1.8,
            "LeftElbow":         2.3, "RightElbow":         2.3,
            "LeftWristRoll":     1.8, "RightWristRoll":     1.8,
            "LeftWristPitch":    1.8, "RightWristPitch":    1.8,
            "LeftWristYaw":      1.8, "RightWristYaw":      1.8,
            "WaistYaw":          1.5, "NotUsedJoint":       1.5,
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
            

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def _initialize_targets_from_current_state(self):
        if self.low_state is None:
            print("[WARN] Cannot initialize targets, low_state not available.")
            return
        for joint in arm_joint_names:
            q = self.low_state.motor_state[joint_mapping[joint]].q
            self.target_positions[joint] = q
        self.target_positions["NotUsedJoint"] = 1.0


    def _gain_update_step(self):
        dt = self.gain_dt_
        with self._gain_lock:
            for j in self.Kp_map.keys():
                # Kp
                dkp = self.Kp_target_map[j] - self.Kp_map[j]
                if dkp != 0.0:
                    step = np.clip(dkp, -self.kp_slew_rate*dt, +self.kp_slew_rate*dt)
                    self.Kp_map[j] = self._clip_gain("kp", j, self.Kp_map[j] + step)
                # Kd
                dkd = self.Kd_target_map[j] - self.Kd_map[j]
                if dkd != 0.0:
                    step = np.clip(dkd, -self.kd_slew_rate*dt, +self.kd_slew_rate*dt)
                    self.Kd_map[j] = self._clip_gain("kd", j, self.Kd_map[j] + step)

    
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
            with self._gain_lock:
                kp_now = self.Kp_map[joint]
                kd_now = self.Kd_map[joint]
            kp = self._clip_gain("kp", joint, kp_now)
            kd = self._clip_gain("kd", joint, kd_now)
            self.low_cmd.motor_cmd[idx].kp = kp
            self.low_cmd.motor_cmd[idx].kd = kd
            # self.low_cmd.motor_cmd[idx].kp = self.Kp_map[joint]
            # self.low_cmd.motor_cmd[idx].kd = self.Kd_map[joint]
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

        # save state values
        if self.logging_enabled:
            self.log_data.append(log_entry)
 

        # Update the weight pasrameter using NotUsedJoint.
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = self.target_positions.get("NotUsedJoint", 1.0)

        # Compute and attach the CRC.
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # Write the command over the high-level arm topic.
        self.armcmd_publisher.Write(self.low_cmd)

    def start_control_loop(self):
        """
        Start a control loop that continuously sends arm commands.
        """
        self.running = True
        self.control_thread = RecurrentThread(
            interval=self.control_dt_, target=self.write_arm_command, name="g1_arm_control_loop"
        )
        self.control_thread.Start()

        # NEW: gain smoother
        self.gain_thread = RecurrentThread(
            interval=self.gain_dt_, target=self._gain_update_step, name="gain_smoother"
        )
        self.gain_thread.Start()

    def stop_control_loop(self):
        """
        Stop the arm control loop.
        """
        self.running = False
        if self.control_thread is not None:
            self.control_thread.Wait()  # Proper way to request loop exit and join
        # self.release_arm_sdk()
        if self.gain_thread is not None:
            self.gain_thread.Wait()
        
        # Stop or join your control thread as appropriate.
    def release_arm_sdk(self):
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.armcmd_publisher.Write(self.low_cmd)
    
    def update_target_positions(self, new_targets: dict):
        """ G1 SDK2, 
        Update the target positions for the arm joints.
        Expect keys from the arm joint list or "NotUsedJoint".
        """
        for joint, target in new_targets.items():
            if joint in self.target_positions:
                self.target_positions[joint] = target
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
    # READOUT HELPER
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
    
    def read_velocity_state(self):
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].dq
                for joint, idx in joint_mapping.items()}
    
    # second layer functions that should be ignored 
    def set_speed_preset(self, name: str):
        # one-layer: no S-curve generator; ignore politely
        print(f"[INFO] (one-layer) set_speed_preset('{name}') ignored.")

    def set_global_speed_scale(self, lam: float):
        print(f"[INFO] (one-layer) set_global_speed_scale({lam}) ignored.")

    def set_joint_speed_caps(self, caps: dict):
        print(f"[INFO] (one-layer) set_joint_speed_caps(...) ignored.")

    # ERGONOMIC HELPER ( MEANING =D efficiency and comfort in the working environment) 
    def enable_ff(self, on: bool):
        self.ff_enabled = bool(on)

    def set_tau_limits(self, tau_lim_map: dict):
        self.tau_limit.update({j: float(v) for j, v in tau_lim_map.items() if j in self.tau_limit})


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
                v_clipped = self._clip_gain("kp", joint, v)
                if v_clipped != v:
                    print(f"[CLIP] kp[{joint}] {v} -> {v_clipped}")
                self.Kp_target_map[joint] = v_clipped

                if self.auto_damping:
                    c = self.kd_per_sqrt_kp_map.get(joint, 0.18)  # fallback to scalar if missing
                    kd_tgt = c * math.sqrt(max(v_clipped, 1e-6))
                    self.Kd_target_map[joint] = self._clip_gain("kd", joint, kd_tgt)

            else:
                v_clipped = self._clip_gain("kd", joint, v)
                if v_clipped != v:
                    print(f"[CLIP] kd[{joint}] {v} -> {v_clipped}")
                self.Kd_target_map[joint] = v_clipped

            updated[joint] = v

        if unknown:
            print(f"[WARN] Unknown joint names: {unknown}")

        return {"updated": updated, "unknown_joints": unknown, "which": which}
    

    def _clip_gain(self, which: str, joint: str, value: float) -> float:
        """
        Clip a proposed gain for `joint` using self.Kp_map_limit / self.Kd_map_limit.
        Accepts either a scalar max or a (lo, hi) tuple per joint. lo defaults to 0.
        """
        which = which.lower()
        limits = self.Kp_map_limit if which == "kp" else self.Kd_map_limit
        lim = limits.get(joint, None)
        if lim is None:
            return value
        if isinstance(lim, (int, float)):
            lo, hi = 0.0, float(lim)
        else:
            lo, hi = float(lim[0]), float(lim[1])
        if value < lo: return lo
        if value > hi: return hi
        return value


# -------------- Test ----------------

def wait_for_state(controller, timeout=5.0):
    t0 = time.time()
    while controller.low_state is None and (time.time() - t0) < timeout:
        time.sleep(0.01)
    if controller.low_state is None:
        raise RuntimeError("LowState not received; check NIC/topic/mode.")

def zero_pose():
    return {j: 0.0 for j in arm_joint_names}

def move_and_hold(controller, targets, hold_s=0.8):
    controller.update_target_positions(targets)
    time.sleep(hold_s)

def read_and_report(controller, tag, targets):
    state = controller.read_motor_state()
    if not state:
        print(f"[{tag}] No joint states yet."); return
    err = {j: abs(targets.get(j, 0.0) - state.get(j, 0.0)) for j in targets}
    mean_err = sum(err.values())/len(err)
    worst = max(err, key=lambda k: err[k])
    print(f"[{tag}] mean|e|={mean_err:.4f} rad, max|e|={err[worst]:.4f} @ {worst}")

def apply_sim_base_gains(controller):
    # Start soft to avoid sim chatter; wrists softer
    base_kp = {j: 35.0 for j in arm_joint_names}
    base_kd = {j: 1.0  for j in arm_joint_names}
    for j in ["LeftWristRoll","LeftWristPitch","LeftWristYaw",
              "RightWristRoll","RightWristPitch","RightWristYaw"]:
        base_kp[j] = 20.0
        base_kd[j] = 0.8
    controller.update_gains(base_kp, which="kp")
    controller.update_gains(base_kd, which="kd")
    print("[INFO] Applied sim base gains.")

def main_basic_io():
    """
    Minimal bring-up test:
      - Initialize DDS + controller (one-layer)
      - Apply sim-friendly joint gains
      - Lerp ZERO -> TARGET -> ZERO
      - Read & print q, dq, tau_est snapshots
      - (Optional) log to CSV
    """
    # --- Config ---
    NIC = os.environ.get("enp2s0", "lo")   # "lo" for loopback; use robot NIC when on hardware
    DDS_DOMAIN = 1
    CTRL_DT = 0.02                         # 50 Hz loop
    SEG_DURATION_S = 3.0
    SEG_STEPS = max(1, int(SEG_DURATION_S / CTRL_DT))

    ZERO = {j: 0.0 for j in arm_joint_names}
    ZERO["NotUsedJoint"] = 1.0

    TARGET = {j: 0.0 for j in arm_joint_names}
    TARGET.update({
        "LeftShoulderPitch":  0.25,
        "LeftShoulderRoll":  -0.15,
        "LeftElbow":         -0.60,
        "RightShoulderPitch": 0.25,
        "RightShoulderRoll":  0.15,
        "RightElbow":        -0.60,
    })

    def lerp_move(ctrl, q_from: dict, q_to: dict, steps: int, dt: float):
        for k in range(1, steps + 1):
            a = k / float(steps)
            interm = {j: (1.0 - a) * q_from.get(j, 0.0) + a * q_to.get(j, 0.0) for j in arm_joint_names}
            ctrl.update_target_positions(interm)
            time.sleep(dt)

    def snapshot_print(ctrl, tag: str):
        q  = ctrl.read_motor_state()
        dq = ctrl.read_velocity_state()
        tq = ctrl.read_torque_state()
        if not q:
            print(f"[{tag}] No state yet.")
            return
        # Simple summaries
        def summarize(d, label):
            vals = [abs(v) for k, v in d.items() if k in arm_joint_names]
            if not vals:
                print(f"  {label}: n/a")
                return
            mean_v = sum(vals) / len(vals)
            mx_k = max(d, key=lambda k: abs(d[k]))
            print(f"  {label}: mean={mean_v:.4f}  max|.|={abs(d[mx_k]):.4f} @ {mx_k}")

        print(f"[SNAPSHOT] {tag}")
        summarize(q,  "q (rad)")
        summarize(dq, "dq (rad/s)")
        summarize(tq, "tau_est (Nm)")

    # --- Bring-up ---
    print("[INFO] DDS init…")
    try:
        # 1 => use loopback NIC name "lo" path; replace with 0,NIC for real NIC when on robot
        ChannelFactoryInitialize(DDS_DOMAIN, NIC)
        time.sleep(0.3)
    except Exception as e:
        print(f"[WARN] DDS init failed on NIC={NIC}: {e}")

    print("[INFO] Controller init (one-layer)…")
    ctrl = UnitreeG1ArmController(control_dt=CTRL_DT, dds_topic="l")
    ctrl.start_control_loop()

    # Optionally enable CSV logging
    # ctrl.enable_logging()  # uncomment if you want a CSV log

    # Wait for LowState to arrive
    wait_for_state(ctrl, timeout=6.0)

    # Apply simulation gains (soft, wrists softer)
    base_kp = {j: 35.0 for j in arm_joint_names}
    base_kd = {j: 1.0  for j in arm_joint_names}
    for j in ["LeftWristRoll","LeftWristPitch","LeftWristYaw",
              "RightWristRoll","RightWristPitch","RightWristYaw"]:
        base_kp[j] = 20.0
        base_kd[j] = 0.8
    ctrl.update_gains(base_kp, which="kp")
    ctrl.update_gains(base_kd, which="kd")
    print("[INFO] Applied sim base gains (Kp=35/Kd=1; wrists 20/0.8).")

    # For a pure PD test, disable FF for now
    ctrl.enable_ff(False)

    # --- Moves ---
    # Move to ZERO (from current)
    print("[STEP] Move to ZERO…")
    current = ctrl.read_motor_state()
    if not current:
        # seed from ZERO if not ready
        current = ZERO.copy()
    lerp_move(ctrl, current, ZERO, SEG_STEPS, CTRL_DT)
    time.sleep(0.5)
    snapshot_print(ctrl, "after ZERO")

    # ZERO -> TARGET
    print("[STEP] ZERO → TARGET…")
    lerp_move(ctrl, ZERO, TARGET, SEG_STEPS, CTRL_DT)
    time.sleep(0.5)
    snapshot_print(ctrl, "at TARGET")

    # TARGET -> ZERO
    print("[STEP] TARGET → ZERO…")
    lerp_move(ctrl, TARGET, ZERO, SEG_STEPS, CTRL_DT)
    time.sleep(0.5)
    snapshot_print(ctrl, "back at ZERO")

    # Save CSV if logging
    ctrl.save_log_to_csv()

    ctrl.stop_control_loop()
    print("[INFO] Basic I/O test done.")

# Allow selecting which main to run
if __name__ == "__main__":
    # Choose one:
    # main_basic_io()              # <- run the new basic I/O test
    # (or keep your previous scripted demo)
    main_basic_io()


#     # --- Config (edit these as needed) ---
#     NIC = os.environ.get("G1_NIC", "enp2s0")   # e.g., "enp2s0" on the robot, "lo" for loopback
#     CTRL_DT = 0.02                             # 50 Hz control loop
#     SEG_DURATION_S = 3.0                       # seconds for each move (ZERO→TARGET, TARGET→ZERO)
#     SEG_STEPS = max(1, int(SEG_DURATION_S / CTRL_DT))

#     # Define your target pose here (radians). Keep values conservative for first test.
#     TARGET = {j: 0.0 for j in arm_joint_names}
#     TARGET.update({
#         "LeftShoulderPitch":  0.25,
#         "LeftShoulderRoll":  -0.15,
#         "LeftElbow":         -0.60,
#         "RightShoulderPitch": 0.25,
#         "RightShoulderRoll":  0.15,
#         "RightElbow":        -0.60,
#         "WaistYaw":           0.00,
#         "NotUsedJoint":       1.00,   # keep your "weight" param at 1.0
#     })

#     def wait_for_lowstate(ctrl, timeout=5.0):
#         t0 = time.time()
#         while ctrl.low_state is None and (time.time() - t0) < timeout:
#             time.sleep(0.01)
#         if ctrl.low_state is None:
#             raise RuntimeError("LowState not received; check NIC/topic/mode.")

#     def get_current_pose(ctrl):
#         state = ctrl.read_motor_state()
#         if not state:
#             # Fallback to zeros if not yet available
#             return {j: 0.0 for j in arm_joint_names}
#         # Ensure NotUsedJoint preserved as well
#         state.setdefault("NotUsedJoint", 1.0)
#         return state

#     def lerp_move(ctrl, q_from: dict, q_to: dict, steps: int):
#         # linear interpolation in joint space
#         for k in range(1, steps + 1):
#             alpha = k / float(steps)
#             interm = {}
#             for j in arm_joint_names:
#                 a = q_from.get(j, 0.0)
#                 b = q_to.get(j, 0.0)
#                 interm[j] = (1.0 - alpha) * a + alpha * b
#             ctrl.update_target_positions(interm)
#             time.sleep(CTRL_DT)

#     print("[INFO] DDS init...")
#     # 0: use given NIC, 1: loopback; pick 0 for real NIC use
#     try:
#         ChannelFactoryInitialize(1, "lo")
#         time.sleep(0.5)
#     except Exception as e:
#         print(f"[WARN] DDS init failed on NIC={NIC}: {e}")

#     print("[INFO] Controller init (low-level)…")
#     ctrl = UnitreeG1ArmController(control_dt=CTRL_DT, dds_topic="l")
#     ctrl.start_control_loop()

#     # Wait for state
#     wait_for_lowstate(ctrl, timeout=6.0)

#     # Disable feedforward for a clean PD test (keeps default Kp/Kd)
#     ctrl.ff_enabled = False

#     # Define ZERO pose
#     ZERO = {j: 0.0 for j in arm_joint_names}
#     ZERO["NotUsedJoint"] = 1.0  # keep your "weight" parameter at 1.0

#     print("[STEP] Move to ZERO (hold)…")
#     lerp_move(ctrl, get_current_pose(ctrl), ZERO, steps=SEG_STEPS)
#     time.sleep(0.5)

#     print("[STEP] ZERO → TARGET…")
#     lerp_move(ctrl, ZERO, TARGET, steps=SEG_STEPS)
#     time.sleep(0.5)

#     print("[STEP] TARGET → ZERO…")
#     lerp_move(ctrl, TARGET, ZERO, steps=SEG_STEPS)
#     time.sleep(0.5)

#     ctrl.stop_control_loop()
#     print("[INFO] Done.")