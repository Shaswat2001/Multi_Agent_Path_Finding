#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')

from marl_planner.agent import MADDPG
from marl_planner.pytorch_model import GaussianPolicyNetwork, PolicyNetwork,CentralizedQNetwork,QNetwork,VNetwork,PhasicPolicyNetwork,PhasicQNetwork,ConstraintNetwork,MultiplierNetwork,SafePolicyNetwork,RealNVP

from pettingzoo.mpe import simple_spread_v3

from marl_planner.replay_buffer.Uniform_RB import ReplayBuffer,VisionReplayBuffer
from marl_planner.replay_buffer.Auxiliary_RB import AuxReplayBuffer
from marl_planner.replay_buffer.Constraint_RB import ConstReplayBuffer,CostReplayBuffer

from marl_planner.exploration.OUActionNoise import OUActionNoise

def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="uam_gazebo",help="Name of OPEN AI environment")
    parser.add_argument("input_shape",nargs="?",type=int,default=[],help="Shape of environment state")
    parser.add_argument("n_actions",nargs="?",type=int,default=[],help="shape of environment action")
    parser.add_argument("max_action",nargs="?",type=float,default=[],help="Max possible value of action")
    parser.add_argument("min_action",nargs="?",type=float,default=[],help="Min possible value of action")

    parser.add_argument("Algorithm",nargs="?",type=str,default="MADDPG",help="Name of RL algorithm")
    parser.add_argument('tau',nargs="?",type=float,default=0.005)
    parser.add_argument('gamma',nargs="?",default=0.99)
    parser.add_argument('actor_lr',nargs="?",type=float,default=0.0001,help="Learning rate of Policy Network")
    parser.add_argument('critic_lr',nargs="?",type=float,default=0.0001,help="Learning rate of the Q Network")

    parser.add_argument("mem_size",nargs="?",type=int,default=100000,help="Size of Replay Buffer")
    parser.add_argument("batch_size",nargs="?",type=int,default=64,help="Batch Size used during training")
    parser.add_argument("n_agents",nargs="?",type=int,default=2,help="Total number of agents in the environment")
    parser.add_argument("n_episodes",nargs="?",type=int,default=50000,help="Total number of episodes to train the agent")
    parser.add_argument("n_batches",nargs="?",type=int,default=10,help="Total number of times the RL needs to be replicated")
    parser.add_argument("target_update",nargs="?",type=int,default=2,help="Iterations to update the target network")
    parser.add_argument("vision_update",nargs="?",type=int,default=5,help="Iterations to update the vision network")
    parser.add_argument("delayed_update",nargs="?",type=int,default=100,help="Iterations to update the second target network using delayed method")
    
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

    args = parser.parse_args("")

    return args

def train(args,env,trainer_dict):

    best_reward = -np.inf
    total_reward = []
    avg_reward_list = []
    os.makedirs("config/saves/rl_rewards/" +args.Environment, exist_ok=True)
    os.makedirs("config/saves/images/" +args.Environment, exist_ok=True)
    
    for i in range(args.n_episodes):

        observation, _ = env.reset()
        reward = 0
        
        while True:

            action = {agent:trainer_dict[agent].choose_action(observation[agent]) for agent in env.agents}
            next_observation,rwd,termination,truncation,info = env.step(action)

            for key,trainer in trainer_dict.items():
                trainer.add(observation[key],action[key],rwd[key],next_observation[key],termination[key])
            
            for trainer in trainer_dict.values():
                trainer.learn(list(trainer_dict.values()))

            reward+=sum(list(rwd.values()))

            if all(list(termination.values())) or all(list(truncation.values())):
                break
                
            observation = next_observation

        total_reward.append(reward)
        avg_reward = np.mean(total_reward[-40:])

        if avg_reward>best_reward and i > 10:
            best_reward=avg_reward
            if args.save_rl_weights:
                print("Weights Saved !!!")
                for trainer in trainer_dict.values():
                    trainer.save(args.Environment)

        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        avg_reward_list.append(avg_reward)

    if args.save_results:
        f = open("config/saves/rl_rewards/" +args.Environment + "/" + args.Algorithm + ".pkl","wb")
        pickle.dump(avg_reward_list,f)
        f.close()
    
    plt.figure()
    plt.title(f"Reward values - {args.Algorithm}")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.plot(avg_reward_list)
    plt.show()

if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()
    trainer_dict = {}

    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env.reset()

    args.n_agents = len(env.agents)
    for an in range(len(env.agents)):
        args.state_size = env.observation_space(env.agents[an]).shape[0]
        args.input_shape = env.observation_space(env.agents[an]).shape[0]
        args.n_actions = env.action_space(env.agents[an]).shape[0]
        args.max_action = env.action_space(env.agents[an]).high
        args.min_action = env.action_space(env.agents[an]).low
    
        if args.Algorithm == "MADDPG":
            trainer = MADDPG.MADDPG(args = args,policy = PolicyNetwork,critic = CentralizedQNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise, agent_num=an)
        
        trainer_dict[env.agents[an]]  = trainer

    train(args,env,trainer_dict)