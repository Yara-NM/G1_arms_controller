# G1 Arms Controller

This repository contains the public thesis-support code for controlling the [Unitree G1 humanoid robot](https://www.unitree.com/g1/) in dual-arm manipulation experiments.

The code was developed as part of the master's thesis:

**Vision-Language-Guided Impedance Control of Humanoid Robot Aimed at Safe Contact-Rich Manipulation**

The repository supports dual-arm motion, forward/inverse kinematics using Pinocchio and CasADi, joint-space impedance control, task-space compliant behavior, camera calibration utilities, and structured logging/plotting tools. The project is ROS-free and was tested with both the Unitree Mujoco simulator and the physical Unitree G1 robot.

This public version includes the main code and selected supporting files needed to understand and reproduce the control pipeline. It does **not** include raw videos, full experimental databases, large datasets, private logs, ZIP archives, or temporary demo folders. :contentReference[oaicite:0]{index=0}
---

## 🧩 System Overview

![System Overview](structure.jpg)

## 📚 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Dependencies Installation](#dependencies-installation)
- [Project Structure](#project-structure)
- [Package Breakdown](#package-breakdown)
- [Thesis Support Files](#thesis-support-files)
- [Database Files](#database-files)
- [Excluded Files](#excluded-files)
- [Demo Files](#demo-files)

---
## Features

- Dual-arm control for the Unitree G1 humanoid robot
- Forward and inverse kinematics using [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) and CasADi
- Joint control through DDS messaging using [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- Joint-space impedance control with stiffness, damping, velocity, and torque-related limits
- Task-space spring-damper compliant behavior
- Gripper control support
- Camera calibration and hand-eye transformation utilities
- Structured logging and plotting for experimental analysis
- Tested with the [Unitree Mujoco simulator](https://github.com/unitreerobotics/Unitree_mujoco)
- ROS-free standalone Python design
  
## Requirements

- Python ≥ 3.8
- NumPy 1.21.5 < 2.0 (required for Pinocchio compatibility)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html)
- CasADi 3.7.0
- SciPy 1.8.0
- unitree_sdk2py (official Unitree Python SDK)
- Unitree Mujoco Simulator (for simulation)

## Dependencies Installation
### 1. Install Unitree SDK2 Python

```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
``` 
### 2. Install Unitree Mujoco (Python-based simulation)

```bash
pip install mujoco
pip install pygame

cd ~
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```
### 3.  Install CasADi, SciPy libraries
```bash
pip install casadi scipy
```
### 4. Install NumPy < 2.0 (required for Pinocchio)
```bash
pip install numpy==1.26.4
```
### 5. Install Pinocchio using robotpkg
Follow this official guide:

👉 https://stack-of-tasks.github.io/pinocchio/download.html

Then, set up your environment:
```bash
echo 'export PATH=/opt/openrobots/bin:$PATH' >> ~/.bashrc
echo 'export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH' >> ~/.bashrc

source ~/.bashrc
```
### 6. Clone This Repository and Set Python Path 
```bash
cd ~
git clone https://github.com/Yara-NM/G1_arms_controller.git
cd  /path to this repo/ 
pip install -e .
```
## 7. Set up the simulator with G1 
Edit the config file:
```bash
nano ~/unitree_mujoco/simulate_python/config.py
```
Change these lines:
```python
ROBOT = "G1"
USE_JOYSTICK = 0
ENABLE_ELASTIC_BAND = True
```
Run the simulator. The Mujoco simulator should start with the Unitree G1 robot:
```bash
cd ~/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```

🕹️ Controls in simulation:

9 — Enable/disable elastic band

7 / 8 — Lift/lower robot

## Project Structure

```text
g1/
├── arms_controller/
│   ├── __init__.py
│   ├── g1_robot_controller.py
│   ├── joint_controller.py
│   ├── joint_controller_2layers.py
│   ├── joint_controller_updated.py
│   ├── g1_ik_solver_with_vis.py
│   ├── cartesian_impedance.py
│   ├── impedance_manager.py
│   ├── plotting.py
│   ├── utils.py
│   └── assets/
│       └── g1/
│           ├── g1_body29_hand14.urdf
│           ├── g1_body29_hand14_fixed.urdf
│           └── meshes/
│
├── camera_calibration/
│   ├── g1_new_camera_intrinsics_1280x720.json
│   └── test_hand_eye_transform_v2.py
│
├── Data_Collection/
│   ├── collect_data.py
│   ├── collect_data2.py
│   ├── collect_data_withoutIK.py
│   └── references_points.py
│
├── task_space_impedance/
│   ├── spring_damper_cartesian.py
│   ├── profiles.py
│   ├── grasp_orientation.py
│   ├── gripper_controller.py
│   ├── plot_task_space_log.py
│   ├── plot_virtual_forces.py
│   ├── Data_Collection_impedance_task_space/
│   └── results/
│
├── impedance_database_g1.json
├── grasping_database_g1.json
├── cartesian_impedance_database_g1.json
├── gripper_controller.py
├── setup.py
├── MANIFEST.in
├── LICENSE
└── README.md
```

---
## Package Breakdown

### `arms_controller/`

Main package for Unitree G1 upper-body control.

Important files:

- `g1_robot_controller.py`  
  High-level interface connecting IK, joint control, logging, and plotting.

- `joint_controller.py`  
  Main joint-space controller using the Unitree low-level command interface.

- `joint_controller_2layers.py`  
  Joint controller with an additional interpolation layer for smoother joint-space trajectories.

- `joint_controller_updated.py`  
  Updated joint controller with task-dependent motion profiles and impedance-related limits.

- `g1_ik_solver_with_vis.py`  
  Pinocchio/CasADi-based inverse kinematics solver with visualization support.

- `cartesian_impedance.py`  
  Task-space impedance module for Cartesian spring-damper behavior.

- `impedance_manager.py`  
  Interface layer connecting joint control, inverse kinematics, and task-space impedance.

- `plotting.py`  
  Tools for saving and visualizing joint position, velocity, and torque logs.

- `utils.py`  
  Utility functions for rotations, quaternions, RPY conversions, and joint-name mapping.
---

## Thesis Support Files

This repository includes selected supporting code and files used in the thesis experiments. The goal is to provide the control logic and curated parameter files, not the full raw experimental archive.

### Camera Calibration

The `camera_calibration/` folder contains selected calibration files used for vision-based experiments.

Included files:

- `g1_new_camera_intrinsics_1280x720.json`  
  Camera intrinsic parameters for the G1 head-mounted camera.

- `test_hand_eye_transform_v2.py`  
  Script for testing the hand-eye transformation between camera and robot frames.

Raw calibration images and generated calibration result folders are not included in this public repository.

### SafeHumanoid Data-Collection Support

The `Data_Collection/` folder contains code used for SafeHumanoid-style data collection and validation.

Included files:

- `collect_data.py`
- `collect_data2.py`
- `collect_data_withoutIK.py`
- `references_points.py`

Only code is included. Raw videos, generated run folders, large result folders, and archived datasets are excluded.

### Task-Space Impedance / HumanoidVLM Support

The `task_space_impedance/` folder contains the main task-space compliant behavior implementation.

It includes:

- Cartesian spring-damper impedance control
- Task-dependent stiffness/damping profiles
- Grasp orientation testing
- Interactive task-space tests
- Gripper control utilities
- Plotting scripts for task-space logs and virtual forces
- Selected result figures used for analysis

The folder `task_space_impedance/results/` includes selected images and plots that support the thesis experiments. Full raw task-space databases, videos, and temporary demo folders are excluded.

---

## Database Files

The root folder includes selected JSON databases used by the retrieval-based impedance and grasping pipeline:

- `impedance_database_g1.json`  
  Joint-space impedance database for task-dependent robot behavior.

- `grasping_database_g1.json`  
  Grasping parameter database for object-specific gripper behavior.

- `cartesian_impedance_database_g1.json`  
  Task-space impedance database for Cartesian compliant behavior.

These files provide curated, validated parameters used to connect semantic scene understanding to physically meaningful control behavior.

---

## Excluded Files

This public repository intentionally excludes:

- Raw experiment videos
- Full databases with camera recordings
- Large result folders
- ZIP archives
- Python cache files
- Temporary demo folders
- Local-only test folders

In particular, the following folders are not part of the public thesis release:

```text
videos/
Data_Collection/results/
Data_Collection/results (1)/
g1_kinematics_example_2004_fixed/
task_space_impedance/MWS_Demo/
task_space_impedance/demo_MWS/
task_space_impedance/Data_Collection_impedance_task_space/database/

---

## Demo Files

The original `demo/` folder is kept only as a lightweight example area. It is not the main thesis-support folder and is not intended to contain raw experiment recordings or large datasets.

For thesis-related examples, see:

- `Data_Collection/` for SafeHumanoid-style data-collection code
- `task_space_impedance/` for task-space compliant behavior and HumanoidVLM-related code


### Remote Instructions

![Remote Instruction](control_instructions.jpg)

---

## Citation

If this repository is used for academic work, please cite the associated thesis or related publications.

```bibtex
@mastersthesis{mahmoud2026vlm_impedance_g1,
  title  = {Vision-Language-Guided Impedance Control of Humanoid Robot Aimed at Safe Contact-Rich Manipulation},
  author = {Mahmoud, Yara},
  school = {Skolkovo Institute of Science and Technology},
  year   = {2026}
}
