import numpy as np
import torch
import os
from marl_planner.network.masoftq_critic import MASoftQCritic
from marl_planner.common.exploration import OUActionNoise
from marl_planner.common.replay_buffer import ReplayBuffer
from marl_planner.common.utils import hard_update,soft_update

class MASoftQ:
    '''
    MASoftQ Algorithm 
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

        _,observation,action,reward,_,next_observation,done = self.replay_buffer.shuffle()

        target_action_list = []
        actions_list = []

        for ai in range(len(self.args.env_agents)):

            agent = self.args.env_agents[ai]
            obs_i = observation[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            next_obs_i = next_observation[:,ai*self.action_space:(ai+1)*self.action_space]

            target_critic_action = self.TargetPolicyNetwork[agent](next_obs_i)
            target_action = self.PolicyNetwork[agent](obs_i)
            target_action_list.append(target_critic_action)
            actions_list.append(target_action)
        
        target_q = self.TargetQNetwork(next_observation,torch.hstack(target_action_list))
        target_v = torch.logsumexp(target_q, dim=1,keepdim=True)
        y = reward.sum(dim = 1, keepdim=True) + self.args.gamma*target_v
        critic_value = self.Qnetwork(observation,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()
        
        critic_value = self.Qnetwork(observation,torch.hstack(actions_list))
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

    def add(self,state,observation,action,reward,next_state,next_observation,done):

        self.replay_buffer.store(state,observation,action,reward,next_state,next_observation,done)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args,reward_type = "ind",action_space="continuous")
        # Exploration Technique
        self.noiseOBJ = {agent:OUActionNoise(mean=np.zeros(self.args.n_actions[agent]), std_deviation=float(0.04) * np.ones(self.args.n_actions[agent])) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}

        self.policy_parameters = []

        for policy in self.PolicyNetwork.values():

            self.policy_parameters += policy.parameters()

        self.PolicyOptimizer = torch.optim.Adam(self.policy_parameters,lr=self.args.actor_lr)

        self.TargetPolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}

        self.Qnetwork = MASoftQCritic(self.args)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = MASoftQCritic(self.args)

        self.network_hard_updates()

    def network_hard_updates(self):

        hard_update(self.TargetQNetwork,self.Qnetwork)
        for agent in self.args.env_agents:
            hard_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent])
    
    def network_soft_updates(self):

        soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)
        for agent in self.args.env_agents:
            soft_update(self.TargetPolicyNetwork[agent],self.PolicyNetwork[agent],self.args.tau)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + f"/masoftq_weights/", exist_ok=True)
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + f"/masoftq_weights/QWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + f"/masoftq_weights/TargetQWeights.pth")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/masoftq_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/masoftq_weights//{agent}/actorWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/masoftq_weights/{agent}/TargetactorWeights.pth")
            

    def load(self,env):

        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/masoftq_weights/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/masoftq_weights/TargetQWeights.pth",map_location=torch.device('cpu')))

        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/masoftq_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/masoftq_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
            