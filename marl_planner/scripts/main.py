#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import copy
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')
from marl_planner.common.arguments import *
from marl_planner.agent import MADDPG, COMA, MAAC, QMIX, MASoftQ, VDN, MATD3, FACMAC, FOP
from marl_planner.network.base_net import DiscreteMLP, DiscreteGaussianNet, ContinuousMLP, RNN, ContGaussianNet
from pettingzoo.mpe import simple_spread_v3, simple_v3, simple_reference_v3

def train(args,env,trainer):

    best_reward = -np.inf
    total_reward = []
    avg_reward_list = []
    os.makedirs("config/saves/rl_rewards/" +args.Environment, exist_ok=True)
    os.makedirs("config/saves/images/" +args.Environment, exist_ok=True)
    sample_added = 0
    for i in range(args.n_episodes):
        observation, _ = env.reset()
        global_reward = 0
        
        while True:

            state = env.state()
            action = trainer.choose_action(observation)

            next_observation,rwd,termination,truncation,info = env.step(action)
            next_state = env.state()
            global_reward += sum(list(rwd.values()))

            if args.Algorithm in ["VDN","QMIX","FACMAC","FOP"]:
                reward = global_reward
            else:
                reward = rwd
            
            sample_added += 1
            if all(list(termination.values())) or all(list(truncation.values())): 
                
                done = {}
                for key in termination.keys():
                    done[key] = True
                
                trainer.add(state,observation,action,reward,next_state,next_observation,done)

                if args.Algorithm in ["COMA"]:
                    if i%args.train_network == 0:
                        trainer.learn()

                break
                
            else:
                trainer.add(state,observation,action,reward,next_state,next_observation,termination)
            
            if args.Algorithm in ["MADDPG","MASoftQ","VDN","MATD3","QMIX","FACMAC"]:
                trainer.learn()

            observation = next_observation

        total_reward.append(global_reward)
        avg_reward = np.mean(total_reward[-40:])

        if avg_reward>best_reward and i > 10:
            best_reward=avg_reward
            if args.save_rl_weights:
                print("Weights Saved !!!")
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

    if args.Algorithm in ["COMA","QMIX","VDN"]:
        args.is_continous = False
    else:
        args.is_continous = True
    
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5,continuous_actions=args.is_continous)
    # env = simple_reference_v3.parallel_env(local_ratio=0.5, max_cycles=25, continuous_actions=args.is_continous)
    # env = simple_v3.parallel_env(continuous_actions=args.is_continous)
    env.reset()

    args = get_env_parameters(args,env)
    
    if args.Algorithm == "MADDPG":
        args = get_maddpg_args(args)
        trainer = MADDPG.MADDPG(args = args,policy = ContinuousMLP)
    elif args.Algorithm == "FACMAC":
        args = get_facmac_args(args)
        trainer = FACMAC.FACMAC(args = args,policy = ContinuousMLP)
    elif args.Algorithm == "MATD3":
        args = get_maddpg_args(args)
        trainer = MATD3.MATD3(args = args,policy = ContinuousMLP)
    elif args.Algorithm == "COMA":
        args = get_coma_args(args)
        trainer = COMA.COMA(args = args,policy = DiscreteMLP)
    elif args.Algorithm == "MAAC":
        args = get_coma_args(args)
        trainer = MAAC.MAAC(args = args,policy = DiscreteGaussianNet)
    elif args.Algorithm == "QMIX":
        args = get_qmix_args(args)
        trainer = QMIX.QMIX(args = args,policy = RNN)
    elif args.Algorithm == "MASoftQ":
        args = get_maddpg_args(args)
        trainer = MASoftQ.MASoftQ(args = args,policy = ContinuousMLP)
    elif args.Algorithm == "VDN":
        args = get_vdn_args(args)
        trainer = VDN.VDN(args = args)
    elif args.Algorithm == "FOP":
        args = get_facmac_args(args)
        trainer = FOP.FOP(args = args,policy = ContGaussianNet)

    train(args,env,trainer)