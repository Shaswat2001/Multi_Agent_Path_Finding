import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update,soft_update
from marl_planner.network.qmix_net import QMixer
from marl_planner.common.replay_buffer import ReplayBuffer
from torch.distributions import Categorical

class QMIX:
    '''
    QMIX Algorithm 
    '''
    def __init__(self,args,policy):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.policy = policy

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

        for ai in range(len(self.args.env_agents)):
            
            agent = self.args.env_agents[ai]

            obs = torch.Tensor(observation[agent])
            qval,self.policy_hidden[:,ai,:] = self.PolicyNetwork[agent](obs,self.policy_hidden[:,ai,:])
            if stage == "training" and np.random.normal() < self.epsilon:

                act = np.random.choice(self.action_space)
                action[agent] = act
            else:
                action[agent] = int(qval.argmax(dim = 1).detach().numpy())
        
        return action
    

    def init_rnn_hidden(self,batch_size):

        self.policy_hidden = torch.zeros((batch_size,self.args.n_agents,self.args.rnn_hidden))
        self.target_policy_hidden = torch.zeros((batch_size,self.args.n_agents,self.args.rnn_hidden))

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        state,observation,action,reward,next_state,next_observation,_ = self.replay_buffer.shuffle()
        q_values = []
        target_q_values = []

        self.init_rnn_hidden(state.shape[0])
        for ai in range(len(self.args.env_agents)):

            agent = self.args.env_agents[ai]

            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            next_obs_i = next_observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape] 
            action_i = action[:,ai].view(-1,1)

            qval,_ = self.PolicyNetwork[agent](obs_i,self.policy_hidden[:,ai,:])
            qval = qval.gather(1,action_i)
            next_qval,_ = self.TargetPolicyNetwork[agent](next_obs_i,self.target_policy_hidden[:,ai,:])
            next_qval,_ = next_qval.max(1,keepdims = True)

            q_values.append(qval)
            target_q_values.append(next_qval)

        q_tot = self.Qmixer(state,torch.hstack(q_values))
        q_tot_target = self.TargetQmixer(next_state,torch.hstack(target_q_values))

        y = reward + self.args.gamma*q_tot_target
        critic_loss = torch.mean(torch.square(y.detach() - q_tot),dim=1)
        self.Optimizer.zero_grad()
        critic_loss.mean().backward()
        self.Optimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

        self.init_rnn_hidden(1)

    def add(self,state,observation,action,reward,next_state,next_observation,done):

        self.replay_buffer.store(state,observation,action,reward,next_state,next_observation,done)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args)
        
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents} 
        self.policy_parameters = []

        for policy in self.PolicyNetwork.values():

            self.policy_parameters += policy.parameters()

        self.TargetPolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents} 

        self.Qmixer = QMixer(self.args)
        self.TargetQmixer = QMixer(self.args)

        self.policy_parameters += self.Qmixer.parameters()

        self.Optimizer = torch.optim.Adam(self.policy_parameters,lr=self.args.critic_lr)
        self.init_rnn_hidden(1)
        self.network_hard_updates()
    
    def network_hard_updates(self):

        hard_update(self.TargetQmixer,self.Qmixer)
        for agent in self.args.env_agents:
            hard_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent])
    
    def network_soft_updates(self):

        soft_update(self.TargetQmixer,self.Qmixer,self.args.tau)
        for agent in self.args.env_agents:
            soft_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent],self.args.tau)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + f"/qmix_weights/", exist_ok=True)

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/qmix_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/qmix_weights//{agent}/actorWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/qmix_weights/{agent}/TargetactorWeights.pth")

    def load(self,env):

        for agent in self.args.env_agents:

            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/qmix_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/qmix_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
