import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update
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
    

    def init_rnn_hidden(self,batch_size):

        self.policy_hidden = torch.zeros((batch_size,self.args.n_agents,self.args.rnn_hidden))
        self.target_policy_hidden = torch.zeros((batch_size,self.args.n_agents,self.args.rnn_hidden))

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        if self.learning_step%self.args.target_update == 0:                
            hard_update(self.TargetQmixer,self.Qmixer)
            hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)

    def add(self,s,action,rwd,next_state,done):

        for agent in self.args.env_agents:
            self.replay_buffer[agent].store(s[agent],action[agent],rwd[agent],next_state[agent],done[agent])

    def get_critic_input(self,id,observation,action):

        observations = torch.hstack(list(observation.values()))
        batch_size = observations.shape[0]
        id = (torch.ones(batch_size)*id).view(-1,1)
        action = torch.hstack(list(action.values()))

        return torch.concatenate((id,observations,action),dim=-1)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args)
        
        self.PolicyNetwork = self.policy(self.args)
        self.TargetPolicyNetwork = self.policy(self.args)

        self.Qmixer = QMixer(self.args)
        self.TargetQmixer = QMixer(self.args)

        self.network_parameters = self.Qmixer.parameters() + self.PolicyNetwork.parameters()

        self.Optimizer = torch.optim.Adam(self.network_parameters,lr=self.args.critic_lr)

        hard_update(self.TargetQmixer,self.Qmixer)
        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + f"/qmix_weights/", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + f"/qmix_weights/actorWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + f"/qmix_weights/TargetactorWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/qmix_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/qmix_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))