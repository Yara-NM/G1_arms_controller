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
    # Right Arm
    "RightShoulderPitch": 22,
    "RightShoulderRoll": 23,
    "RightShoulderYaw": 24,
    "RightElbow": 25,
    "RightWristRoll": 26,
    "RightWristPitch": 27,
    "RightWristYaw": 28,
    "WaistYaw": 12,
    # weight
    "NotUsedJoint": 29
}

arm_joint_names = [
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWristYaw", "NotUsedJoint", "WaistYaw"
]
weak_motors_indices = list(joint_mapping.values())


class UnitreeG1ArmController:
    def __init__(self, control_dt=0.02, controller_layers_dt=0.05, results_dir=None, dds_topic="l"):
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

        # --- NEW: Continuous motion limits & presets ---
        self.global_speed_scale = 1.0  # λ multiplier for v/a/j
        # Default per-joint caps (rad/s, rad/s^2, rad/s^3) — conservative
        self.v_max = {j: math.radians(40.0) for j in arm_joint_names}
        self.a_max = {j: math.radians(200.0) for j in arm_joint_names}
        self.j_max = {j: math.radians(2000.0) for j in arm_joint_names}

        # Separate softer caps for wrists vs shoulders (optional fine-tune)
        for j in ["LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
                  "RightWristRoll", "RightWristPitch", "RightWristYaw"]:
            self.v_max[j] = math.radians(35.0)
            self.a_max[j] = math.radians(160.0)
            self.j_max[j] = math.radians(1600.0)

        # Runtime state for S-curve generator
        self._speed_lock = threading.Lock()

        self._q_cmd = {j: 0.0 for j in arm_joint_names}
        self._v_cmd = {j: 0.0 for j in arm_joint_names}
        self._a_cmd = {j: 0.0 for j in arm_joint_names}

        # Presets (deg units here; converted on set)
        self.speed_presets = {
            "human_near":     {"v": 15.0, "a": 80.0,  "j": 800.0,  "lambda": 0.8},
            "fragile":        {"v": 10.0, "a": 60.0,  "j": 600.0,  "lambda": 0.9},
            "normal":         {"v": 40.0, "a": 200.0, "j": 2000.0, "lambda": 1.0},
            "fast_autonomy":  {"v": 55.0, "a": 300.0, "j": 2500.0, "lambda": 1.0},
        }

        # --- Gains (as in your version) ---
        self.joint_speed_deg_s = {}        # kept for API compatibility (unused by new generator)
        self._min_step_rad = math.radians(0.5)  # kept for compatibility (unused by new generator)

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

        # Gain targets & smoothing
        self.Kp_target_map = dict(self.Kp_map)
        self.Kd_target_map = dict(self.Kd_map)
        self.kp_slew_rate = 20.0
        self.kd_slew_rate = 1.0
        self.auto_damping = True
        self.kd_per_sqrt_kp_map = {
            j: (self.Kd_map[j] / math.sqrt(max(self.Kp_map[j], 1e-6)))
            for j in self.Kp_map.keys()
        }
        self.gain_dt_ = 0.02
        self._gain_lock = threading.Lock()
        self.gain_thread = None

        # Feedforward torque (kept)
        self.tau_ff_map = {j: 0.0 for j in arm_joint_names}
        self.ff_enabled = True
        self.ff_gain = 0.8
        self.ff_alpha = 0.9
        self.tau_limit = {
            "LeftShoulderPitch": 25.0, "RightShoulderPitch": 25.0,
            "LeftShoulderRoll":  25.0, "RightShoulderRoll":  25.0,
            "LeftShoulderYaw":   20.0, "RightShoulderYaw":   20.0,
            "LeftElbow":         20.0, "RightElbow":         20.0,
            "LeftWristRoll":     10.0, "RightWristRoll":     10.0,
            "LeftWristPitch":    10.0, "RightWristPitch":    10.0,
            "LeftWristYaw":      10.0, "RightWristYaw":      10.0,
        }

        # Kp/Kd limits
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

        # initialize msgs
        self.dds_topic_type = dds_topic
        self._init_low_cmd()
        self._init_topics(dds_topic)
        self.control_thread = None
        self.outer_controller_thread = None

        self._initialize_targets_from_current_state()

    # ---------- Topics / State ----------

    def _init_low_cmd(self):
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
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def _initialize_targets_from_current_state(self):
        if self.low_state is None:
            print("[WARN] Cannot initialize targets, low_state not available yet.")
            return
        for joint in arm_joint_names:
            q = self.low_state.motor_state[joint_mapping[joint]].q
            self.target_positions[joint] = q
            self.pending_targets[joint] = q
            # NEW: initialize generator state at current q
            self._q_cmd[joint] = q
            self._v_cmd[joint] = 0.0
            self._a_cmd[joint] = 0.0
        self.pending_targets["NotUsedJoint"] = 1.0
        self.target_positions["NotUsedJoint"] = 1.0

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    # ---------- Speed Presets / Caps ----------

    def set_speed_preset(self, name: str):
        p = self.speed_presets.get(name)
        if not p:
            print(f"[WARN] Unknown preset '{name}'. Available: {list(self.speed_presets)}")
            return
        with self._speed_lock:
            for j in arm_joint_names:
                self.v_max[j] = math.radians(p["v"])
                self.a_max[j] = math.radians(p["a"])
                self.j_max[j] = math.radians(p["j"])
            self.global_speed_scale = float(p.get("lambda", 1.0))
        print(f"[INFO] Speed preset -> {name}")

    def set_global_speed_scale(self, lam: float):
        self.global_speed_scale = max(0.05, float(lam))
        print(f"[INFO] Global speed scale λ -> {self.global_speed_scale:.2f}")

    def set_joint_speed_caps(self, caps: dict):
        """caps: {joint: {'v': deg_s, 'a': deg_s2, 'j': deg_s3}}"""
        for j, c in caps.items():
            if j not in arm_joint_names:
                continue
            if "v" in c: self.v_max[j] = math.radians(float(c["v"]))
            if "a" in c: self.a_max[j] = math.radians(float(c["a"]))
            if "j" in c: self.j_max[j] = math.radians(float(c["j"]))

    # ---------- Control Threads ----------

    def start_control_loop(self):
        self.running = True
        self.control_thread = RecurrentThread(
            interval=self.control_dt_, target=self.write_arm_command, name="g1_arm_control_loop"
        )
        self.control_thread.Start()
        self.outer_controller_thread = RecurrentThread(
            interval=self.controller_layers_dt_, target=self.stepwise_update_target_positions, name="continuous_profile_updater"
        )
        self.outer_controller_thread.Start()
        self.gain_thread = RecurrentThread(
            interval=self.gain_dt_, target=self._gain_update_step, name="gain_smoother"
        )
        self.gain_thread.Start()

    def stop_control_loop(self):
        self.running = False
        if self.control_thread is not None:
            self.control_thread.Wait()
        if self.gain_thread is not None:
            self.gain_thread.Wait()

    # ---------- Command Write ----------

    def write_arm_command(self):
        if self.low_state is None:
            return

        log_entry = {"time": time.time()}
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.low_state.mode_machine

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

            if self.ff_enabled:
                lim = self.tau_limit.get(joint, 10.0)
                self.low_cmd.motor_cmd[idx].tau = np.clip(self.ff_gain * self.tau_ff_map[joint], -lim, +lim)
            else:
                self.low_cmd.motor_cmd[idx].tau = 0.0

            if self.logging_enabled:
                log_entry[f"{joint}_target"] = target_q
                log_entry[f"{joint}_pos"] = self.low_state.motor_state[idx].q
                log_entry[f"{joint}_vel"] = self.low_state.motor_state[idx].dq
                log_entry[f"{joint}_tau"] = self.low_state.motor_state[idx].tau_est
                log_entry[f"{joint}_step_target"] = self.target_positions[joint]
                log_entry[f"{joint}_final_target"] = self.pending_targets[joint]

        if self.logging_enabled:
            self.log_data.append(log_entry)

        # Update the weight parameter using NotUsedJoint.
        self.low_cmd.motor_cmd[joint_mapping["NotUsedJoint"]].q = self.target_positions.get("NotUsedJoint", 1.0)

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.armcmd_publisher.Write(self.low_cmd)

    # ---------- NEW: Continuous S-curve-ish profile ----------

    def stepwise_update_target_positions(self):
        """
        Continuous generator with jerk-limited accel, accel-limited vel:
          - Ramp acceleration toward +/-a_max with jerk j_max
          - Clip acceleration by a_max and velocity by v_max
          - Integrate to get q_cmd and write into target_positions
        """
        dt = float(self.controller_layers_dt_)
        if dt <= 0.0:
            return

        with self._speed_lock:
            lam = self.global_speed_scale
            v_lim_map = {j: self.v_max[j] * lam for j in arm_joint_names}
            a_lim_map = {j: self.a_max[j] * lam for j in arm_joint_names}
            j_lim_map = {j: self.j_max[j] * lam for j in arm_joint_names}

        for j in arm_joint_names:
            v_lim = v_lim_map[j]; a_lim = a_lim_map[j]; j_lim = j_lim_map[j]
            q_des = self.pending_targets[j]
            q     = self._q_cmd[j]
            v     = self._v_cmd[j]
            a     = self._a_cmd[j]

            err = q_des - q
            if abs(err) < 1e-6 and abs(v) < 1e-6:
                # Snap to target when effectively there
                self._q_cmd[j] = q_des
                self.target_positions[j] = q_des
                self._v_cmd[j] = 0.0
                self._a_cmd[j] = 0.0
                continue

            sign = 1.0 if err > 0.0 else -1.0

            # Braking heuristic: if stopping distance >= remaining distance, start braking
            v_abs = abs(v)
            stop_dist = 0.5 * (v_abs * v_abs) / max(a_lim, 1e-6)
            want_brake = (stop_dist >= abs(err)) and (v_abs > 1e-6)

            a_target = (-a_lim if v > 0 else a_lim) * sign if want_brake else (a_lim * sign)

            # Jerk-limit acceleration toward a_target
            da_max = j_lim * dt
            a = a + np.clip(a_target - a, -da_max, +da_max)
            a = np.clip(a, -a_lim, +a_lim)

            # Accel-limit velocity
            v = v + a * dt
            v = np.clip(v, -v_lim, +v_lim)

            # Integrate position
            q_next = q + v * dt

            # Clamp if we're about to cross the target
            if (q - q_des) * (q_next - q_des) <= 0.0 and abs(err) < max(v_lim * dt * 0.6, 1e-4):
                q_next = q_des
                v = 0.0
                a = 0.0

            # Commit
            self._q_cmd[j] = q_next
            self._v_cmd[j] = v
            self._a_cmd[j] = a
            self.target_positions[j] = q_next

    # ---------- Gains ----------

    def _gain_update_step(self):
        dt = self.gain_dt_
        with self._gain_lock:
            for j in self.Kp_map.keys():
                dkp = self.Kp_target_map[j] - self.Kp_map[j]
                if dkp != 0.0:
                    step = np.clip(dkp, -self.kp_slew_rate * dt, +self.kp_slew_rate * dt)
                    self.Kp_map[j] = self._clip_gain("kp", j, self.Kp_map[j] + step)
                dkd = self.Kd_target_map[j] - self.Kd_map[j]
                if dkd != 0.0:
                    step = np.clip(dkd, -self.kd_slew_rate * dt, +self.kd_slew_rate * dt)
                    self.Kd_map[j] = self._clip_gain("kd", j, self.Kd_map[j] + step)

    def update_gains(self, gains: dict, which: str):
        if which is None:
            raise ValueError("Specify which='kp' or 'kd'.")
        which = which.lower()
        if which not in ("kp", "kd"):
            raise ValueError("Argument 'which' must be 'kp' or 'kd'.")

        updated, unknown = {}, []
        for joint, val in gains.items():
            if joint not in self.Kp_map:
                unknown.append(joint)
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                print(f"[WARN] Gain for {joint} must be a number. Skipped.")
                continue
            if not np.isfinite(v) or v < 0.0:
                print(f"[WARN] Invalid gain for {joint}. Skipped.")
                continue

            if which == "kp":
                v_clipped = self._clip_gain("kp", joint, v)
                if v_clipped != v: print(f"[CLIP] kp[{joint}] {v} -> {v_clipped}")
                self.Kp_target_map[joint] = v_clipped
                if self.auto_damping:
                    c = self.kd_per_sqrt_kp_map.get(joint, 0.18)
                    kd_tgt = c * math.sqrt(max(v_clipped, 1e-6))
                    self.Kd_target_map[joint] = self._clip_gain("kd", joint, kd_tgt)
            else:
                v_clipped = self._clip_gain("kd", joint, v)
                if v_clipped != v: print(f"[CLIP] kd[{joint}] {v} -> {v_clipped}")
                self.Kd_target_map[joint] = v_clipped

            updated[joint] = v

        if unknown:
            print(f("[WARN] Unknown joint names: {unknown}"))
        return {"updated": updated, "unknown_joints": unknown, "which": which}

    def _clip_gain(self, which: str, joint: str, value: float) -> float:
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

    # ---------- Readouts / Logging ----------

    def read_motor_state(self):
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[joint_mapping[joint]].q for joint in arm_joint_names}

    def read_torque_state(self):
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].tau_est for joint, idx in joint_mapping.items()}

    def enable_logging(self, filename=None):
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

    # ---------- Target API ----------

    def update_target_positions(self, new_targets: dict):
        for joint, target in new_targets.items():
            if joint in self.pending_targets:
                self.pending_targets[joint] = float(target)
            else:
                print(f"Warning: {joint} not found in target positions.")

    def update_feedforward_torque(self, tau_map: dict, alpha=None):
        a = self.ff_alpha if alpha is None else float(alpha)
        for j, v in tau_map.items():
            if j in self.tau_ff_map:
                try:
                    v = float(v)
                except Exception:
                    continue
                self.tau_ff_map[j] = (1.0 - a) * self.tau_ff_map[j] + a * v

    # ---------- Time Estimate (trapezoid/triangle per joint) ----------

    def estimate_total_motion_time(self):
        """
        Estimate time to reach current pending_targets under per-joint v/a caps.
        Uses trapezoidal/triangular profile approximation per joint.
        """
        if self.low_state is None:
            print("[WARN] Low state not yet received.")
            return 0.0

        lam = self.global_speed_scale
        t_max = 0.0
        for j in arm_joint_names:
            q_cur = self.low_state.motor_state[joint_mapping[j]].q
            q_des = self.pending_targets.get(j, q_cur)
            d = abs(q_des - q_cur)
            if d <= 0.0:
                continue

            v = self.v_max[j] * lam
            a = self.a_max[j] * lam
            if v <= 1e-9 or a <= 1e-9:
                continue

            t_acc = v / a
            d_acc = 0.5 * a * t_acc * t_acc

            if d >= 2 * d_acc:
                # trapezoid
                d_cruise = d - 2 * d_acc
                t = 2 * t_acc + d_cruise / v
            else:
                # triangle (peak velocity lower than v_max)
                t = 2 * math.sqrt(d / a)
            if t > t_max:
                t_max = t
        # add a small settling margin
        return t_max + 0.3


# ===== Helpers =====
def wait_for_state(controller, timeout=5.0):
    t0 = time.time()
    while controller.low_state is None and (time.time() - t0) < timeout:
        time.sleep(0.01)
    if controller.low_state is None:
        raise RuntimeError("LowState not received")

def zero_pose():
    return {j: 0.0 for j in arm_joint_names}

def move_and_hold(ctrl, targets, extra_hold=0.8):
    ctrl.update_target_positions(targets)
    t_est = ctrl.estimate_total_motion_time()
    time.sleep(t_est + extra_hold)

def sleep_for_gain_ramp(ctrl, joint, new_kp=None, new_kd=None, margin=0.3):
    with ctrl._gain_lock:
        kp_now = ctrl.Kp_map[joint]; kd_now = ctrl.Kd_map[joint]
        kp_rate = ctrl.kp_slew_rate; kd_rate = ctrl.kd_slew_rate
    t_kp = abs((new_kp - kp_now) / kp_rate) if new_kp is not None else 0.0
    t_kd = abs((new_kd - kd_now) / kd_rate) if new_kd is not None else 0.0
    time.sleep(max(t_kp, t_kd) + margin)


# ===== Demo / Test =====
# def main():
    # print("[INFO] DDS init...")
    # ChannelFactoryInitialize(1, "lo")
    # time.sleep(0.5)

    # print("[INFO] Controller init...")
    # ctrl = UnitreeG1ArmController(control_dt=0.02,
    #                               controller_layers_dt=0.05,
    #                               dds_topic="l")
    # ctrl.start_control_loop()
    # # ctrl.enable_logging()
    # wait_for_state(ctrl)
    # ctrl.ff_enabled = False

    # # Soft sim-like base gains (kept from your example)
    # base_kp = {j: 35.0 for j in arm_joint_names}
    # base_kd = {j: 1.0  for j in arm_joint_names}
    # for j in ["LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    #           "RightWristRoll", "RightWristPitch", "RightWristYaw"]:
    #     base_kp[j] = 20.0; base_kd[j] = 0.8
    # ctrl.update_gains(base_kp, "kp"); ctrl.update_gains(base_kd, "kd")
    # time.sleep(0.1)

    # # Keep waist/weight steady; go ZERO
    # ctrl.update_target_positions({"WaistYaw": 0.0, "NotUsedJoint": 1.0})
    # ZERO = zero_pose()

    # # Define a simple motion to test speed differences (left elbow flex)
    # STEP = zero_pose()
    # STEP["LeftElbow"] = -0.6

    # # Test the four presets on the SAME motion
    # presets = ["human_near", "fragile", "normal", "fast_autonomy"]

    # print("\n[STEP] Move to ZERO at 'normal' preset...")
    # ctrl.set_speed_preset("normal")
    # move_and_hold(ctrl, ZERO, extra_hold=0.8)

    # for name in presets:
    #     print(f"\n[TEST] Preset -> {name}")
    #     ctrl.set_speed_preset(name)
    #     # Optionally add a global scaler here:
    #     # ctrl.set_global_speed_scale(0.9)
    #     move_and_hold(ctrl, STEP, extra_hold=1.0)
    #     move_and_hold(ctrl, ZERO, extra_hold=0.8)

    # # Quick demo of changing λ without changing preset
    # print("\n[TEST] normal preset with λ=0.5 (half speed)")
    # ctrl.set_speed_preset("normal")
    # ctrl.set_global_speed_scale(0.5)
    # move_and_hold(ctrl, STEP, extra_hold=1.0)
    # move_and_hold(ctrl, ZERO, extra_hold=0.8)

    # ctrl.stop_control_loop()
    # # ctrl.save_log_to_csv()
    # print("[INFO] Done.")

# def main():
#     print("[INFO] DDS init...")
#     ChannelFactoryInitialize(1, "lo")
#     time.sleep(0.5)

#     print("[INFO] Controller init...")
#     ctrl = UnitreeG1ArmController(control_dt=0.02,
#                                   controller_layers_dt=0.05,
#                                   dds_topic="l")
#     ctrl.start_control_loop()
#     wait_for_state(ctrl)
#     ctrl.ff_enabled = False

#     # Keep waist/weight steady
#     ctrl.update_target_positions({"WaistYaw": 0.0, "NotUsedJoint": 1.0})

#     # We will NOT change λ in this test; presets only.
#     ctrl.set_global_speed_scale(1.0)

#     ZERO = zero_pose()

#     # Four bilateral poses (safe ranges), same as before
#     POSES = [
#         {   # Pose A: elbows flex, slight shoulder pitch
#             "LeftShoulderPitch":  +0.25,  "RightShoulderPitch": +0.25,
#             "LeftElbow":          -0.60,  "RightElbow":         -0.60,
#             "LeftWristYaw":       +0.15,  "RightWristYaw":      -0.15,
#             "WaistYaw":            0.0,   "NotUsedJoint":        1.0,
#         },
#         {   # Pose B: reach forward a bit, tiny wrist pitch
#             "LeftShoulderPitch":  +0.35,  "RightShoulderPitch": +0.35,
#             "LeftElbow":          -0.40,  "RightElbow":         -0.40,
#             "LeftWristPitch":     +0.12,  "RightWristPitch":    +0.12,
#             "WaistYaw":            0.0,   "NotUsedJoint":        1.0,
#         },
#         {   # Pose C: small ab/adduction with roll
#             "LeftShoulderRoll":   +0.20,  "RightShoulderRoll":  -0.20,
#             "LeftElbow":          -0.50,  "RightElbow":         -0.50,
#             "LeftWristRoll":      +0.10,  "RightWristRoll":     -0.10,
#             "WaistYaw":            0.0,   "NotUsedJoint":        1.0,
#         },
#         {   # Pose D: a bit more reach + yaw
#             "LeftShoulderPitch":  +0.40,  "RightShoulderPitch": +0.40,
#             "LeftShoulderYaw":    +0.15,  "RightShoulderYaw":   -0.15,
#             "LeftElbow":          -0.55,  "RightElbow":         -0.55,
#             "WaistYaw":            0.0,   "NotUsedJoint":        1.0,
#         },
#     ]

#     PRESETS = ["human_near", "fragile", "normal", "fast_autonomy"]

#     # Move to ZERO once to start (use current/default preset)
#     print("\n[STEP] Move to ZERO (startup)")
#     move_and_hold(ctrl, ZERO, extra_hold=0.8)

#     # Sweep presets; for each preset run: ZERO -> POSE -> ZERO
#     for name, pose in zip(PRESETS, POSES):
#         print(f"\n[RUN] Preset -> {name}")
#         ctrl.set_speed_preset(name)     # sets per-joint v/a/j + λ for the preset
#         ctrl.set_global_speed_scale(1.0)  # keep λ fixed (explicit)

#         # Go from current ZERO → preset pose
#         move_and_hold(ctrl, pose, extra_hold=1.0)

#         # Return to ZERO at the same preset
#         move_and_hold(ctrl, ZERO, extra_hold=0.8)

#     ctrl.stop_control_loop()
#     print("[INFO] Done.")

def main():
    print("[INFO] DDS init...")
    ChannelFactoryInitialize(1, "lo")
    time.sleep(0.5)

    print("[INFO] Controller init...")
    ctrl = UnitreeG1ArmController(control_dt=0.02,
                                  controller_layers_dt=0.05,
                                  dds_topic="l")
    ctrl.start_control_loop()
    wait_for_state(ctrl)
    ctrl.ff_enabled = False

    # Decouple Kd from Kp so you can set them independently
    ctrl.auto_damping = False

    # Baseline gains (will be ramped by the running gain thread)
    BASE_KP = {j: 35.0 for j in arm_joint_names}
    BASE_KD = {j: 1.0  for j in arm_joint_names}
    # for j in ["LeftWristRoll","LeftWristPitch","LeftWristYaw",
    #           "RightWristRoll","RightWristPitch","RightWristYaw"]:
    #     BASE_KP[j] = 20.0
    #     BASE_KD[j] = 0.8
    ctrl.update_gains(BASE_KP, "kp")
    ctrl.update_gains(BASE_KD, "kd")

    # Keep waist/weight steady
    ctrl.update_target_positions({"WaistYaw": 0.0, "NotUsedJoint": 1.0})
    ctrl.set_global_speed_scale(1.0)  # λ stays 1.0; presets drive v/a/j

    ZERO = zero_pose()

    # Four bilateral poses (safe ranges)
    POSES = [
        {"LeftShoulderPitch": +0.25, "RightShoulderPitch": +0.25,
         "LeftElbow": -0.60, "RightElbow": -0.60,
         "LeftWristYaw": +0.15, "RightWristYaw": -0.15,
         "WaistYaw": 0.0, "NotUsedJoint": 1.0},
        {"LeftShoulderPitch": +0.35, "RightShoulderPitch": +0.35,
         "LeftElbow": -0.40, "RightElbow": -0.40,
         "LeftWristPitch": +0.12, "RightWristPitch": +0.12,
         "WaistYaw": 0.0, "NotUsedJoint": 1.0},
        {"LeftShoulderRoll": +0.20, "RightShoulderRoll": -0.20,
         "LeftElbow": -0.50, "RightElbow": -0.50,
         "LeftWristRoll": +0.10, "RightWristRoll": -0.10,
         "WaistYaw": 0.0, "NotUsedJoint": 1.0},
        {"LeftShoulderPitch": +0.40, "RightShoulderPitch": +0.40,
         "LeftShoulderYaw": +0.15, "RightShoulderYaw": -0.15,
         "LeftElbow": -0.55, "RightElbow": -0.55,
         "WaistYaw": 0.0, "NotUsedJoint": 1.0},
    ]

    # Per-task independent gain tweaks (optional)
    # You can leave any dict empty to keep baseline.
    TASKS = [
        {
            "preset": "human_near",
            # a bit more damping at low speeds, slightly softer Kp on elbows
            "kp": {"LeftElbow": 30.0, "RightElbow": 30.0},
            "kd": {"LeftElbow": 1.2,  "RightElbow": 1.2},
        },
        {
            "preset": "fragile",
            # keep kp modest; add wrist damping
            "kp": {},
            "kd": {"LeftWristPitch": 1.2, "RightWristPitch": 1.2},
        },
        {
            "preset": "normal",
            # back to baseline (if you had changed earlier)
            "kp": BASE_KP,
            "kd": BASE_KD,
        },
        {
            "preset": "fast_autonomy",
            # slightly higher kp on shoulders, a touch more kd to curb ringing
            "kp": {"LeftShoulderPitch": 45.0, "RightShoulderPitch": 45.0,
                   "LeftShoulderRoll":  45.0, "RightShoulderRoll":  45.0},
            "kd": {"LeftShoulderPitch": 1.2, "RightShoulderPitch": 1.2,
                   "LeftShoulderRoll":  1.2, "RightShoulderRoll":  1.2},
        },
    ]

    print("\n[STEP] Move to ZERO (startup)")
    move_and_hold(ctrl, ZERO, extra_hold=0.8)
    input ("Press Enter to start..")

    # Sweep tasks: for each, set preset (speed envelope), then independently set Kp/Kd targets
    for pose, task in zip(POSES, TASKS):
        name = task["preset"]
        print(f"\n[RUN] Preset -> {name} (independent gains)")
        ctrl.set_speed_preset(name)   # changes v/a/j only

        if task.get("kp"):
            ctrl.update_gains(task["kp"], "kp")  # gain thread ramps these
        if task.get("kd"):
            ctrl.update_gains(task["kd"], "kd")

        # Move to pose and back to ZERO at this preset + gains
        move_and_hold(ctrl, pose, extra_hold=1.0)
        move_and_hold(ctrl, ZERO, extra_hold=0.8)

    ctrl.stop_control_loop()
    print("[INFO] Done.")



if __name__ == "__main__":
    main()
