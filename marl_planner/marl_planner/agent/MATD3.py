import numpy as np
import torch
import os
from marl_planner.network.maddpg_critic import MADDPGCritic
from marl_planner.common.exploration import OUActionNoise
from marl_planner.common.replay_buffer import ReplayBuffer
from marl_planner.common.utils import hard_update,soft_update

class MATD3:
    '''
    MATD3 Algorithm 
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

        for ai in range(len(self.args.env_agents)):

            _,observation,action,reward,_,next_observation,done = self.replay_buffer.shuffle()
            agent = self.args.env_agents[ai]

            reward_i = reward[:,ai].view(-1,1)
            done_i = done[:,ai].view(-1,1)
            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]

            target_action_list = []
            for aj in range(len(self.args.env_agents)):

                agt = self.args.env_agents[aj]
                next_obs_i = next_observation[:,aj*self.obs_shape:(aj+1)*self.obs_shape]

                target_critic_action = self.TargetPolicyNetwork[agt](next_obs_i)
                target_action_list.append(target_critic_action)

            target1 = self.TargetQNetwork1[agent](next_observation,torch.hstack(target_action_list))
            target2 = self.TargetQNetwork2[agent](next_observation,torch.hstack(target_action_list))
            y = reward_i + self.args.gamma*torch.minimum(target1,target2)*(1-done_i)
            critic_value = self.Qnetwork1[agent](observation,action)
            critic_loss = torch.mean(torch.square(y.detach() - critic_value),dim=1)
            self.QOptimizer1[agent].zero_grad()
            critic_loss.mean().backward()
            self.QOptimizer1[agent].step()

            critic_value = self.Qnetwork2[agent](observation,action)
            critic_loss = torch.mean(torch.square(y.detach() - critic_value),dim=1)
            self.QOptimizer2[agent].zero_grad()
            critic_loss.mean().backward()
            self.QOptimizer2[agent].step()

            action[:,ai*self.action_space:(ai+1)*self.action_space] = self.PolicyNetwork[agent](obs_i)
            critic_value = self.Qnetwork1[agent](observation,action)
            actor_loss = -critic_value.mean()
            self.PolicyOptimizer[agent].zero_grad()
            actor_loss.mean().backward()
            self.PolicyOptimizer[agent].step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

    def add(self,state,observation,action,reward,next_state,next_observation,done):

        self.replay_buffer.store(state,observation,action,reward,next_state,next_observation,done)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args,reward_type="ind",action_space="continuous")
        # Exploration Technique
        self.noiseOBJ = {agent:OUActionNoise(mean=np.zeros(self.args.n_actions[agent]), std_deviation=float(0.3) * np.ones(self.args.n_actions[agent])) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}
        self.PolicyOptimizer = {agent:torch.optim.Adam(self.PolicyNetwork[agent].parameters(),lr=self.args.actor_lr) for agent in self.args.env_agents}
        self.TargetPolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}

        self.Qnetwork1 = {agent:MADDPGCritic(self.args,agent) for agent in self.args.env_agents}
        self.QOptimizer1 = {agent:torch.optim.Adam(self.Qnetwork1[agent].parameters(),lr=self.args.critic_lr) for agent in self.args.env_agents}
        self.TargetQNetwork1 = {agent:MADDPGCritic(self.args,agent) for agent in self.args.env_agents}

        self.Qnetwork2 = {agent:MADDPGCritic(self.args,agent) for agent in self.args.env_agents}
        self.QOptimizer2 = {agent:torch.optim.Adam(self.Qnetwork2[agent].parameters(),lr=self.args.critic_lr) for agent in self.args.env_agents}
        self.TargetQNetwork2 = {agent:MADDPGCritic(self.args,agent) for agent in self.args.env_agents}

        self.network_hard_updates()

    def network_hard_updates(self):

        for agent in self.args.env_agents:
            hard_update(self.TargetQNetwork1[agent],self.Qnetwork1[agent])
            hard_update(self.TargetQNetwork2[agent],self.Qnetwork2[agent])
            hard_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent])
    
    def network_soft_updates(self):

        for agent in self.args.env_agents:
            soft_update(self.TargetQNetwork1[agent],self.Qnetwork1[agent],self.args.tau)
            soft_update(self.TargetQNetwork2[agent],self.Qnetwork2[agent],self.args.tau)
            soft_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent],self.args.tau)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/matd3_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/matd3_weights//{agent}/actorWeights.pth")
            torch.save(self.Qnetwork1[agent].state_dict(),"config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/QWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/TargetactorWeights.pth")
            torch.save(self.TargetQNetwork1[agent].state_dict(),"config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/TargetQWeights.pth")

    def load(self,env):

        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/matd3_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.Qnetwork1[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/QWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
            self.TargetQNetwork1[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/matd3_weights/{agent}/TargetQWeights.pth",map_location=torch.device('cpu')))