import json
from pathlib import Path
# Generated Output for follow_surface.png:
# Task: Follow the curve

# ----------------------------------------------------
# Generated Output for massage.jpg:
# Task: Massage

# ----------------------------------------------------
# Generated Output for place.png:
# Task: Place hard and gentle objects

# ----------------------------------------------------
# Generated Output for stab.jpg:
# Task: Poke the fruit/vegetable

# ----------------------------------------------------

# ---------- SCENARIO DEFINITIONS (edit / add here) ----------

g1_1_clear_workspace_follow_surface = {
    "scenario_id": "g1_clear_workspace_cartisian_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles follow a surface with right end effector",
    "task_type": "maintain a contact in z direction",


    "impedance_parameters": {
        "left_ee": {
            "K": [6.0, 6.0, 3.0],
            "D": [2.0, 2.0, 2.0],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [6.0, 6.0, 3.0],
            "D": [2.0, 2.0, 2.0],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_2_clear_workspace_massage = {
    "scenario_id": "g1_clear_workspace_cartisian_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. robot holdeing massage ball. robot apply force with right end effector",
    "task_type": "apply concentrated force in z direction",


    "impedance_parameters": {
        "left_ee": {
            "K": [6.0, 6.0, 3.0],
            "D": [2.0, 2.0, 2.0],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [3.0, 3.0, 5.0],
            "D": [2.0, 2.0, 3.0],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_3_clear_workspace_different_objects_simultaneously = {
    "scenario_id": "g1_clear_workspace_cartisian_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. robot holdeing sauce bottel in the left hand and an egg in the right hand. the robot place both object on the table with different behaviour",
    "task_type": "place an object on the table",


    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [4.0, 4.0, 2.0],
            "D": [3.0, 3.0, 1.0],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_4_clear_workspace_poke_tomato_with_fork = {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. robot holdeing a fork in the right hand and will poke a fruit or vegetable with it",
    "task_type": "place an object on the table",

    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [3.0, 3.0, 2.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_5_clear_workspace_pick_up_soft_ball= {
    "scenario_id": "g1_clear_workspace_cartisian_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp a ball with the correct angle and pick it up ",
    "task_type": "place an object on the table",

    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_6_clear_workspace_pick_up_egg= {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp an egg with the correct angle and pick it up ",
    "task_type": "place an object on the table",

    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_7_clear_workspace_pick_up_sause= {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp small bottel of sauce with the correct angle and pick it up ",
    "task_type": "place an object on the table",

    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_8_clear_workspace_pick_up_soft_toys= {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp a soft toy with the correct angle and pick it up ",
    "task_type": "place an object on the table",


    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }

}

g1_9_clear_workspace_pick_up_small_plant= {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp a small plant with the correct angle and pick it up ",
    "task_type": "place an object on the table",


    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

g1_10_clear_workspace_pick_up_cloth= {
    "scenario_id": "g1_clear_workspace_impedance",
    "scenario": "G1 robot in clear workspace with no obstacles. the right hand will move to grasp a piece of cloth with the correct angle and pick it up ",
    "task_type": "place an object on the table",


    "impedance_parameters": {
        "left_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        },
        "right_ee": {
            "K": [5.0, 5.0, 4.0],
            "D": [2.0, 2.0, 1.5],
            "M": [1.5, 1.5, 1.5]
        }
    }
}

grasp_1 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold no object. fully closed gripper",

        "grippers": {
            "angle_close_deg": 200.0,
            "angle_open_deg": 0.0
        },
}

grasp_2 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold egg",

        "grippers": {
            "angle_close_deg": 60.0,
            "angle_open_deg": 0.0
        },
}

grasp_3 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold small bottle, sauce bottle, tooth paste",

        "grippers": {
            "angle_close_deg": 80.0,
            "angle_open_deg": 0.0
        },
}

grasp_4 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold small soft toy, squishy, deformable object",

        "grippers": {
            "angle_close_deg": 130.0,
            "angle_open_deg": 0.0
        },
}
grasp_4 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold spiece of cloth, wiping cloth, fabric",

        "grippers": {
            "angle_close_deg": 195.0,
            "angle_open_deg": 0.0
        },
}

grasp_5 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold small planet, leaves, small bot",

        "grippers": {
            "angle_close_deg": 70.0,
            "angle_open_deg": 0.0
        },
}

grasp_6 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold small planet, leaves, small bot",

        "grippers": {
            "angle_close_deg": 70.0,
            "angle_open_deg": 0.0
        },
}

grasp_6 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold small soft squishable deformable ball",

        "grippers": {
            "angle_close_deg": 70.0,
            "angle_open_deg": 0.0
        },
}

grasp_7 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold a fork, spoon, knife",

        "grippers": {
            "angle_close_deg": 190.0,
            "angle_open_deg": 0.0
        },
}

grasp_8 = {

        "scenario_id": "g1_gripper configeration",
        "task_type": "manipulate / grasp / hold a massage ball, tool, hamdle",

        "grippers": {
            "angle_close_deg": 180.0,
            "angle_open_deg": 0.0
        },
}


# Add more scenarios here...


# ---------- COLLECT & SAVE ----------

database_impedance = [
    g1_1_clear_workspace_follow_surface,
    g1_2_clear_workspace_massage,
    g1_3_clear_workspace_different_objects_simultaneously,
    g1_4_clear_workspace_poke_tomato_with_fork,
    g1_5_clear_workspace_pick_up_soft_ball,
    g1_6_clear_workspace_pick_up_egg,
    g1_7_clear_workspace_pick_up_sause,
    g1_8_clear_workspace_pick_up_soft_toys,
    g1_9_clear_workspace_pick_up_small_plant,
    g1_10_clear_workspace_pick_up_cloth
]

database_gripper_config = [

    grasp_1,
    grasp_2,
    grasp_3,
    grasp_4,
    grasp_5,
    grasp_6,
    grasp_7,
    grasp_8

]
output_path = Path("cartesian_impedance_database_g1.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(database_impedance, f, indent=4)

print(f"Saved {len(database_impedance)} scenarios to {output_path}")

output_path = Path("grasping_database_g1.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(database_gripper_config, f, indent=4)

print(f"Saved {len(database_gripper_config)} scenarios to {output_path}")