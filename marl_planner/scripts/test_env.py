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
from pettingzoo.mpe import simple_spread_v3, simple_v3

def train(args,env,trainer):

    trainer.load(args.Environment)
    observation, _ = env.reset()        
    while True:

        action = trainer.choose_action(observation,"testing")

        next_observation,rwd,termination,truncation,info = env.step(action)
        if all(list(termination.values())) or all(list(truncation.values())):
            break
            
        observation = next_observation

if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()

    if args.Algorithm in ["COMA","QMIX","VDN"]:
        args.is_continous = False
    else:
        args.is_continous = True

    # env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5,continuous_actions=args.is_continous,render_mode="human")
    env = simple_v3.parallel_env(continuous_actions=args.is_continous,render_mode="human",max_cycles=100)
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