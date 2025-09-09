import time, os, math
from arms_controller import G1RobotArmController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import numpy as np

# === Set Results Directory Relative to This Script ===
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Initialize DDS communication ===
try: 
    ChannelFactoryInitialize(0, "enp2s0")    # ==Simulation NOT SUPPORTED==
    # Give DDS a moment to set up
    time.sleep(0.5) 

except: 
    print ("DDS communication failed")


# === Initialize Robot Controller ===
robot = G1RobotArmController(ctrl_dt=0.02 ,
                             ctrl_2l_dt= 0.2 ,
                             results_dir= RESULTS_DIR,
                              mode= "l" )

# Creates a timestamped CSV file in 'results/'
# robot.start_logging()  
robot.start()  # Starts control loop and resets arms to zero
time.sleep(2)



# === Reset arms to zero position ===
robot.reset_arms()
time.sleep(6)  # Wait for arms to reach zero

# === Get current right end-effector pose using FK ===
right_pos, right_rot = robot.get_R_ee_pose()
left_pos, left_rot = robot.get_L_ee_pose()
left_tf = np.eye(4)
left_tf[:3, :3] = left_rot
left_tf[:3, 3] = left_pos
print("Current right end-effector position (xyz):", right_pos)
print("Current right end-effector rotation matrix:\n", right_rot)

# === Draw a circle in the z-y plane (centered at current pos, radius 0.2 m) ===
radius = 0.2  # 20 cm
center = right_pos.copy()
num_points = 200
theta = np.linspace(0, 2*np.pi, num_points)

for t in theta:
    # Circle in zy-plane (y = center_y + r*cos(t), z = center_z + r*sin(t))
    target_pos = center.copy()
    target_pos[1] = center[1] + radius * np.cos(t)  # y
    target_pos[2] = center[2] + radius * np.sin(t)  # z

    # Use the current orientation, only change position
    target_tf = np.eye(4)
    target_tf[:3, :3] = right_rot
    target_tf[:3, 3] = target_pos

    robot.move_arms(left_tf, target_tf)
    time.sleep(0.1)  # ~20Hz update

# === Reset arms to zero position ===
robot.reset_arms()
time.sleep(6)  # Wait for arms to reach zero
right_pos, right_rot = robot.get_R_ee_pose()
left_pos, left_rot = robot.get_L_ee_pose()

# -- Move 1: Both arms forward (x+20cm) --
right_pos = right_pos + np.array([0.2, 0.0, 0.0])
left_pos = left_pos + np.array([0.2, 0.0, 0.0])

right_tf = np.eye(4)
right_tf[:3, :3] = right_rot
right_tf[:3, 3] = right_pos
left_tf = np.eye(4)
left_tf[:3, :3] = left_rot
left_tf[:3, 3] = left_pos

robot.move_arms(left_tf, right_tf)
time.sleep(3)

# -- Move 2: Both arms up (z+20cm), rotate -45° around y --
right_pos = right_pos + np.array([0.0, 0.0, 0.2])
left_pos = left_pos + np.array([0.0, 0.0, 0.2])

rot_y = math.radians(-45)
R_y = np.array([
    [math.cos(rot_y), 0, math.sin(rot_y)],
    [0, 1, 0],
    [-math.sin(rot_y), 0, math.cos(rot_y)]
])

right_rot = right_rot @ R_y
left_rot = left_rot @ R_y

right_tf = np.eye(4)
right_tf[:3, :3] = right_rot
right_tf[:3, 3] = right_pos
left_tf = np.eye(4)
left_tf[:3, :3] = left_rot
left_tf[:3, 3] = left_pos

robot.move_arms(left_tf, right_tf)
time.sleep(3)
# -- Move 2: Both arms up (z+20cm), rotate -45° around y --
right_pos = right_pos + np.array([0.0, 0.0, 0.2])
left_pos = left_pos + np.array([0.0, 0.0, 0.2])

rot_y = math.radians(-45)
R_y = np.array([
    [math.cos(rot_y), 0, math.sin(rot_y)],
    [0, 1, 0],
    [-math.sin(rot_y), 0, math.cos(rot_y)]
])

right_rot = right_rot @ R_y
left_rot = left_rot @ R_y
robot.m
right_tf = np.eye(4)
right_tf[:3, :3] = right_rot
right_tf[:3, 3] = right_pos
left_tf = np.eye(4)
left_tf[:3, :3] = left_rot
left_tf[:3, 3] = left_pos

robot.move_arms(left_tf, right_tf)
time.sleep(3)

# -- Move 3: Both arms out in y (right -20cm, left +20cm), rotate ±45° around z --
right_pos = right_pos + np.array([0.0, -0.2, 0.0])
left_pos = left_pos + np.array([0.0, 0.2, 0.0])

rot_z = math.radians(45)
R_z_pos = np.array([
    [math.cos(rot_z), -math.sin(rot_z), 0],
    [math.sin(rot_z),  math.cos(rot_z), 0],
    [0, 0, 1]
])
R_z_neg = np.array([
    [math.cos(-rot_z), -math.sin(-rot_z), 0],
    [math.sin(-rot_z),  math.cos(-rot_z), 0],
    [0, 0, 1]
])

right_rot = right_rot @ R_z_neg
left_rot = left_rot @ R_z_pos

right_tf = np.eye(4)
right_tf[:3, :3] = right_rot
right_tf[:3, 3] = right_pos
left_tf = np.eye(4)
left_tf[:3, :3] = left_rot
left_tf[:3, 3] = left_pos

robot.move_arms(left_tf, right_tf)
time.sleep(3)
# -- Move 4: Return to zero --
robot.reset_arms()
time.sleep(6)
print("Sequence done.")
# === Stop controlling and logging ===
robot.stop()
# robot.save_log()
print("Demo complete. Log saved.")
