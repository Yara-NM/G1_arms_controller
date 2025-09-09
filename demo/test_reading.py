from arms_controller import G1RobotArmController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import time

# === Initialize DDS communication ===
try: 
    ChannelFactoryInitialize(0, "enp2s0")    # ==Simulation NOT SUPPORTED==
    # Give DDS a moment to set up
    time.sleep(0.5) 

except: 
    print ("DDS communication failed")


g1 = G1RobotArmController(ctrl_dt=0.02,
                          ctrl_2l_dt= 0.2,
                          results_dir= None,
                          mode= 'l',
                          visualize= False
                          )
time.sleep (0.5)
pos, rot = g1.get_L_ee_pose()
print (pos)
print (rot)