#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')

from marl_planner.agent import DDPG
from marl_planner.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,SafePolicyNetwork,RealNVP,FeatureExtractor

from marl_planner.replay_buffer.Uniform_RB import ReplayBuffer,VisionReplayBuffer
from marl_planner.replay_buffer.Auxiliary_RB import AuxReplayBuffer
from marl_planner.replay_buffer.Constraint_RB import ConstReplayBuffer,CostReplayBuffer

from marl_planner.exploration.OUActionNoise import OUActionNoise

from marl_planner.environment.GazeboEnv.BaseGazeboUAVPosEnv import BaseGazeboUAVPosEnv
from marl_planner.environment.GazeboEnv.BaseGazeboUAVPosObsEnv import BaseGazeboUAVPosObsEnv
from marl_planner.environment.GazeboEnv.BaseGazeboUAVVelEnv import BaseGazeboUAVVelEnv
from marl_planner.environment.GazeboEnv.BaseGazeboUAVVelObsEnv import BaseGazeboUAVVelObsEnv
from marl_planner.environment.GazeboEnv.BaseGazeboUAVVelObsEnvSimp import BaseGazeboUAVVelObsEnvSimp

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uam_gazebo",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="TD3",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")
    parser.add_argument('mult_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the LAG constraint")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("n_batches",nargs="?",type=int,default=10,help="Total number of times the RL needs to be replicated")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    parser.add_argument("enable_vision",nargs="?",type=bool,default=False,help="Whether you want to integrate sensor data")
    
    # SOFT ACTOR PARAMETERS
    parser.add_argument("temperature",nargs="?",type=float,default=0.2,help="Entropy Parameter")
    parser.add_argument("log_std_min",nargs="?",type=float,default=np.log(1e-4),help="")
    parser.add_argument("log_std_max",nargs="?",type=float,default=np.log(4),help="")
    parser.add_argument("aux_step",nargs="?",type=int,default=8,help="How often the auxiliary update is performed")
    parser.add_argument("aux_epoch",nargs="?",type=int,default=6,help="How often the auxiliary update is performed")
    parser.add_argument("target_entropy_beta",nargs="?",type=float,default=-3,help="")
    parser.add_argument("target_entropy",nargs="?",type=float,default=-3,help="")

    # MISC VARIABLES 
    parser.add_argument("save_rl_weights",nargs="?",type=bool,default=True,help="save reinforcement learning weights")
    parser.add_argument("save_results",nargs="?",type=bool,default=True,help="Save average rewards using pickle")

    # USL 
    parser.add_argument("eta",nargs="?",type=float,default=0.05,help="USL eta")
    parser.add_argument("delta",nargs="?",type=float,default=0.1,help="USL delta")
    parser.add_argument("Niter",nargs="?",type=int,default=20,help="Iterations")
    parser.add_argument("cost_discount",nargs="?",type=float,default=0.99,help="Iterations")
    parser.add_argument("kappa",nargs="?",type=float,default=5,help="Iterations")
    parser.add_argument("cost_violation",nargs="?",type=int,default=20,help="Save average rewards using pickle")

    # Safe RL parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("safe_max_action",nargs="?",type=float,default=[],help="Max possible value of safe action")
    parser.add_argument("safe_min_action",nargs="?",type=float,default=[],help="Min possible value of safe action")

    # Environment Teaching parameters
    parser.add_argument("safe_iterations",nargs="?",type=int,default=5,help="Iterations to run Safe RL once engaged")
    parser.add_argument("teach_alg",nargs="?",type=str,default="alp_gmm",help="How to change the environment")

    # Environment parameters List
    parser.add_argument("max_obstacles",nargs="?",type=int,default=10,help="Maximum number of obstacles need in the environment")
    parser.add_argument("obs_region",nargs="?",type=float,default=6,help="Region within which obstacles should be added")

    # ALP GMM parameters
    parser.add_argument('gmm_fitness_fun',nargs="?", type=str, default="aic")
    parser.add_argument('warm_start',nargs="?", type=bool, default=False)
    parser.add_argument('nb_em_init',nargs="?", type=int, default=1)
    parser.add_argument('min_k', nargs="?", type=int, default=2)
    parser.add_argument('max_k', nargs="?", type=int, default=11)
    parser.add_argument('fit_rate', nargs="?", type=int, default=250)
    parser.add_argument('alp_buffer_size', nargs="?", type=int, default=500)
    parser.add_argument('random_task_ratio', nargs="?", type=int, default=0.2)
    parser.add_argument('alp_max_size', nargs="?", type=int, default=None)

    args = parser.parse_args("")

    return args

def train(args,env,agent):

    best_reward = -np.inf
    total_reward = []
    total_safe_reward = []
    avg_reward_list = []
    constraint_violation = []
    os.makedirs("config/saves/rl_rewards/" +args.Environment, exist_ok=True)
    os.makedirs("config/saves/images/" +args.Environment, exist_ok=True)
    
    ep_len = 0
    constraint_broke = 0
    for i in range(args.n_episodes):

        s = env.reset()
        reward = 0
        safe_reward = 0
        
        # for _ in range(200):
        while True:
            # s = s.reshape(1,s.shape[0])
            if env.check_contact: 
                env.max_time = 20
                action = -env.action_max + (2*env.action_max)*np.random.sample(size=(1,3))
            else:
                action = agent.choose_action(s)
            # action = agent.choose_action(s)
            next_state,rwd,done,info = env.step(action)
            ep_len+=1

            agent.add(s,action,rwd,next_state,done)
            agent.learn()
            reward+=rwd
            # print(next_state)
            if done:
                break
                
            s = next_state

        total_reward.append(reward)
        avg_reward = np.mean(total_reward[-40:])
        total_safe_reward.append(safe_reward)
        avg_safe_reward = np.mean(total_safe_reward[-40:])
        constraint_violation.append(constraint_broke/ep_len)
        if avg_reward>best_reward and i > 10:
            best_reward=avg_reward
            if args.save_rl_weights:
                print("Weights Saved !!!")
                agent.save(args.Environment)

        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        print("Episode * {} * Constraint percentage is ==> {}".format(i,constraint_broke/ep_len))
        print("Episode * {} * Avg Safe Reward is ==> {}".format(i, avg_safe_reward))
        avg_reward_list.append(avg_reward)

    if args.save_results:
        list_cont_rwd = [avg_reward_list,constraint_violation,avg_safe_reward]
        f = open("config/saves/rl_rewards/" +args.Environment + "/" + args.Algorithm + ".pkl","wb")
        pickle.dump(list_cont_rwd,f)
        f.close()
    
    plt.figure()
    plt.subplot(211)
    plt.title(f"Constraint Violation (%) - {args.Algorithm}")
    plt.ylabel("Constraint Violation (%)")
    plt.plot(constraint_violation)

    plt.subplot(212)
    plt.title(f"Reward values - {args.Algorithm}")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.plot(avg_reward_list)
    plt.show()

if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()

    param_bound = dict()
    param_bound["n_obstacles"] = [0,args.max_obstacles]
    param_bound["obs_centre"] = [0,args.obs_region,3]

    if "uav_gazebo" == args.Environment:
        env = BaseGazeboUAVPosEnv()
    elif "uav_obs_gazebo" == args.Environment:
        env = BaseGazeboUAVPosObsEnv()
    elif "uav_vel_gazebo" == args.Environment:
        env = BaseGazeboUAVVelEnv()
    elif "uav_vel_obs_gazebo" == args.Environment:
        env = BaseGazeboUAVVelObsEnv()
    elif "uav_vel_obs_gazebo1" == args.Environment:
        env = BaseGazeboUAVVelObsEnvSimp()
    
    args.state_size = env.state_size
    args.input_shape = env.state_size
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.min_action = env.action_space.low
    args.safe_max_action = env.safe_action_max
    args.safe_min_action = -env.safe_action_max


    if args.Algorithm == "DDPG":
        agent = DDPG.DDPG(args = args,policy = PolicyNetwork,critic = QNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)

    train(args,env,agent)