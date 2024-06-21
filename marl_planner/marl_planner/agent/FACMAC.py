import numpy as np
import torch
import os
from marl_planner.network.facmac_critic import FACMACCritic
from marl_planner.common.exploration import OUActionNoise
from marl_planner.network.qmix_net import QMixer
from marl_planner.common.replay_buffer import ReplayBuffer
from marl_planner.common.utils import hard_update,soft_update

class FACMAC:
    '''
    FACMAC Algorithm 
    '''
    def __init__(self,args,policy):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        self.obs_shape = self.args.input_shape[self.args.env_agents[0]]
        self.action_space = self.args.n_actions[self.args.env_agents[0]]
        self.policy = policy

        self.reset()

    def choose_action(self,observation,stage="training"):

        action = {}
        for agent in self.args.env_agents:

            state = torch.Tensor(observation[agent])
            if stage == "training":
                
                act_n = self.PolicyNetwork[agent](state).detach().numpy()
                act_n += self.noiseOBJ[agent]()
            else:
                act_n = self.TargetPolicyNetwork[agent](state).detach().numpy()

            act_n = np.clip(act_n,self.args.min_action[agent],self.args.max_action[agent])
        
            action[agent] = act_n

        return action

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        state,observation,action,reward,next_state,next_observation,_ = self.replay_buffer.shuffle()
        q_values = []
        target_q_values = []

        for ai in range(len(self.args.env_agents)):

            agent = self.args.env_agents[ai]
            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            next_obs_i = next_observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            action_i = action[:,ai*self.action_space:(ai+1)*self.action_space]

            target_critic_action = self.TargetPolicyNetwork[agent](next_obs_i)

            qval = self.Qnetwork[agent](obs_i,action_i)

            next_qval = self.TargetQNetwork[agent](next_obs_i,target_critic_action)

            q_values.append(qval)
            target_q_values.append(next_qval)

        q_tot = self.Qmixer(state,torch.hstack(q_values))
        q_tot_target = self.TargetQmixer(next_state,torch.hstack(target_q_values))

        y = reward + self.args.gamma*q_tot_target
        critic_loss = torch.mean(torch.square(y.detach() - q_tot),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        q_values = []

        for ai in range(len(self.args.env_agents)):

            agent = self.args.env_agents[ai]
            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]

            critic_action = self.PolicyNetwork[agent](obs_i)
            qval = self.Qnetwork[agent](obs_i,critic_action)
            q_values.append(qval)

        q_tot = self.Qmixer(state,torch.hstack(q_values))
        actor_loss = -q_tot.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

    def add(self,state,observation,action,reward,next_state,next_observation,done):

        self.replay_buffer.store(state,observation,action,reward,next_state,next_observation,done)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args,reward_type="global",action_space="continuous")
        # Exploration Technique
        self.noiseOBJ = {agent:OUActionNoise(mean=np.zeros(self.args.n_actions[agent]), std_deviation=float(0.3) * np.ones(self.args.n_actions[agent])) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}
        self.TargetPolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}

        self.Qnetwork = {agent:FACMACCritic(self.args,agent) for agent in self.args.env_agents}
        self.TargetQNetwork = {agent:FACMACCritic(self.args,agent) for agent in self.args.env_agents}

        self.Qmixer = QMixer(self.args)
        self.TargetQmixer = QMixer(self.args)

        self.qnet_parameters = []
        self.policy_parameters = []
        for qnet in self.Qnetwork.values():

            self.qnet_parameters += qnet.parameters()

        self.qnet_parameters += self.Qmixer.parameters()

        for policy in self.PolicyNetwork.values():

            self.policy_parameters += policy.parameters()

        self.QOptimizer = torch.optim.Adam(self.qnet_parameters,lr=self.args.critic_lr)
        self.PolicyOptimizer = torch.optim.Adam(self.policy_parameters,lr=self.args.actor_lr)
        self.network_hard_updates()

    def network_hard_updates(self):

        hard_update(self.TargetQmixer,self.Qmixer)
        for agent in self.args.env_agents:
            hard_update(self.TargetQNetwork[agent],self.Qnetwork[agent])
            hard_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent])
    
    def network_soft_updates(self):
        
        soft_update(self.TargetQmixer,self.Qmixer,self.args.tau)
        for agent in self.args.env_agents:
            soft_update(self.TargetQNetwork[agent],self.Qnetwork[agent],self.args.tau)
            soft_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent],self.args.tau)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maddpg_weights//{agent}/actorWeights.pth")
            torch.save(self.Qnetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/QWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/TargetactorWeights.pth")
            torch.save(self.TargetQNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/TargetQWeights.pth")

    def load(self,env):

        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maddpg_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.Qnetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/QWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
            self.TargetQNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}/TargetQWeights.pth",map_location=torch.device('cpu')))