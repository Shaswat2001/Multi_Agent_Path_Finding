import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import sys
from baselines.common import plot_util as pu

def tsplot(ax, data,name,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.4, **kw)
    ax.plot(x,pu.smooth(est,radius=40),label=name,**kw)
    ax.margins(x=0)

if __name__ == "__main__":

    fig, axmain = plt.subplots(ncols=1, sharey=True)

    if len(sys.argv) == 1:
        environment = "uav_vel_obs_gazebo1"
    else:
        environment = sys.argv[1]

    files = glob.glob(f"config/saves/rl_rewards/{environment}/*.pkl")
    for name in files:
        f = open(name,"rb")
        reward = pickle.load(f)[0]
        reward = np.array(reward).reshape(len(reward),-1)
        f.close()

        reward = np.array(reward)
        ax = tsplot(axmain, reward,name.split("/")[-1][:-4])

    agent_name = [name.split("/")[-1] for name in files]
    Agents = [name[:-4] for name in agent_name]
    print(Agents)
    axmain.legend(loc="upper left")
    axmain.set_xlabel("Episodes")
    axmain.set_ylabel("Average Reward")

    plt.savefig("rewards.png")