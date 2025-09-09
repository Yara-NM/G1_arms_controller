import time, os
import sys
import numpy as np
from pathlib import Path
from g1_ik_solver import G1_IK_Arms




if __name__ == "__main__":
    ik_solver = G1_IK_Arms(visualize= False)
    q_dict, tau_dict, ok = ik_solver.ik_both_pose(sys.argv[0], sys.argv[1], sys.argv[2])
    out = Path.home() / "message_from_robot.txt"
    out.write_text(str(q_dict) + "\n", encoding="utf-8")