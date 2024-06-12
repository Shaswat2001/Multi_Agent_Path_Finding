#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')

from marl_planner.agent import MADDPG, COMA
from marl_planner.pytorch_model import PolicyNetwork,CentralizedQNetwork,QNetwork,VNetwork, DiscretePolicyNetwork, DiscreteQNetwork

from pettingzoo.mpe import simple_spread_v3

from marl_planner.replay_buffer.Uniform_RB import ReplayBuffer, DiscreteReplayBuffer
from marl_planner.exploration.OUActionNoise import OUActionNoise

def get_parameters(args,env):

    args.state_size = {}
    args.input_shape = {}
    args.n_actions = {}
    args.max_action = {}
    args.min_action = {}
    args.env_agents = env.agents
    args.n_agents = len(env.agents)
    for agent in env.agents:
        args.state_size[agent] = env.observation_space(agent).shape[0]
        args.input_shape[agent] = env.observation_space(agent).shape[0]
        args.n_actions[agent] = env.action_space(agent).shape[0]
        args.max_action[agent] = env.action_space(agent).high
        args.min_action[agent] = env.action_space(agent).low


def build_parse():

    parser = argparse.ArgumentParser(description="RL Algorithm Variables")

    parser.add_argument("Environment",nargs="?",type=str,default="simple_spread",help="Name of OPEN AI environment")
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
    parser.add_argument("env_agents",nargs="?",type=list,default=[],help="Name of environment agents")
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

def train(args,env,trainer):

    trainer.load(args.Environment)
    observation, _ = env.reset()        
    while True:

        action = trainer.choose_action(observation)

        next_observation,rwd,termination,truncation,info = env.step(action)

        if all(list(termination.values())) or all(list(truncation.values())):
            break
            
        observation = next_observation

if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()

    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True,render_mode="human")
    env.reset()

    get_parameters(args,env)
    
    if args.Algorithm == "MADDPG":
        trainer = MADDPG.MADDPG(args = args,policy = PolicyNetwork,critic = CentralizedQNetwork,replayBuff = ReplayBuffer,exploration = OUActionNoise)
    if args.Algorithm == "COMA":
        trainer = COMA.COMA(args = args,policy = DiscretePolicyNetwork,critic = DiscreteQNetwork,replayBuff = DiscreteReplayBuffer)

    train(args,env,trainer)