import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import sys 

if __name__ == "__main__":

    fig, axmain = plt.subplots(ncols=1, sharey=True)

    if len(sys.argv) == 1:
        environment = "uav_vel_obs_gazebo1"
    else:
        environment = sys.argv[1]
    
    files = glob.glob(f"config/saves/rl_rewards/{environment}/*.pkl")

    for name in files:
        f = open(name,"rb")
        values = pickle.load(f)
        f.close()

        reward = values[0]
        constraint = values[1]
        reward = np.array(reward).reshape(len(reward),-1)
        print(f"The agent is : {name.split('/')[-1][:-4]}")
        print(f"the maximum reward is : {np.max(np.array(reward))}")
        print(f"the average reward is : {np.average(np.array(reward))}")
        print(f"the constraint voilation is : {np.mean(np.array(constraint))}")