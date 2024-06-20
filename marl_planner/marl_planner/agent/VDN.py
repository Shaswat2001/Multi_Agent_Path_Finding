import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update,soft_update
from marl_planner.network.vdn_net import VDNCritic, VDNMixer
from marl_planner.common.replay_buffer import ReplayBuffer


class VDN:
    '''
    VDN Algorithm 
    '''
    def __init__(self,args):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning

        self.obs_shape = self.args.input_shape[self.args.env_agents[0]]
        self.action_space = self.args.n_actions[self.args.env_agents[0]]
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min

        self.epsilon_decay = (self.epsilon - self.epsilon_min)/50000

        self.reset()

    def choose_action(self,observation,stage="training"):
        if self.learning_step < 50000:
            self.epsilon -= self.epsilon_decay

            self.epsilon = max(self.epsilon,self.epsilon_min)
        else:
            self.epsilon = self.epsilon_min

        action = {}
        for agent in self.args.env_agents:
            
            state = torch.Tensor(observation[agent])
            qval = self.PolicyNetwork[agent](state)

            if stage == "training" and np.random.normal() < self.epsilon:

                act = np.random.choice(self.action_space)
                action[agent] = act
            else:
                action[agent] = int(qval.argmax(dim = 0).detach().numpy())
        
        return action

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        _,observation,action,reward,_,next_observation,_ = self.replay_buffer.shuffle()
        q_values = []
        target_q_values = []

        for ai in range(len(self.args.env_agents)):

            agent = self.args.env_agents[ai]

            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            next_obs_i = next_observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape] 
            action_i = action[:,ai].view(-1,1)

            qval = self.PolicyNetwork[agent](obs_i).gather(1,action_i)
            next_qval,_ = self.TargetPolicyNetwork[agent](next_obs_i).max(1,keepdims = True)

            q_values.append(qval)
            target_q_values.append(next_qval)

        q_tot = self.VDNMixer(torch.hstack(q_values))
        q_tot_target = self.VDNMixer(torch.hstack(target_q_values))

        y = reward + self.args.gamma*q_tot_target
        critic_loss = torch.mean(torch.square(y.detach() - q_tot),dim=1)
        self.PolicyOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

    def add(self,state,observation,action,reward,next_state,next_observation,done):

        self.replay_buffer.store(state,observation,action,reward,next_state,next_observation,done)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args)
                
        self.PolicyNetwork = {agent:VDNCritic(self.args,agent) for agent in self.args.env_agents} 
        self.policy_parameters = []

        for policy in self.PolicyNetwork.values():

            self.policy_parameters += policy.parameters()

        self.PolicyOptimizer = torch.optim.Adam(self.policy_parameters,lr=self.args.actor_lr)
        self.TargetPolicyNetwork = {agent:VDNCritic(self.args,agent) for agent in self.args.env_agents}

        self.VDNMixer = VDNMixer() 
        self.TargetVDNMixer = VDNMixer() 

        self.network_hard_updates()
    
    def network_hard_updates(self):

        for agent in self.args.env_agents:
            hard_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent])
    
    def network_soft_updates(self):

        for agent in self.args.env_agents:
            soft_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent],self.args.tau)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/vdn_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/vdn_weights/{agent}/actorWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/vdn_weights/{agent}/TargetactorWeights.pth")

    def load(self,env):
        
        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/vdn_weights/{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/vdn_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
        
