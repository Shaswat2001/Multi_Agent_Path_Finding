import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update,soft_update
from marl_planner.network.attention_critic import AttentionCritic
from marl_planner.common.replay_buffer import ReplayBuffer


class MAAC:
    '''
    MAAC Algorithm 
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
                
                act_n = self.PolicyNetwork[agent](state,sample=True).action.detach().numpy()
            else:
                act_n = self.TargetPolicyNetwork[agent](state).action.detach().numpy()
        
            action[agent] = act_n

        return action

    def learn(self,agents):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        state_n = []
        action_n = []
        reward_n = []
        next_state_n = []
        done_n = []

        for i in range(len(agents)):
            state,action,reward,next_state,done = agents[i].replay_buffer.shuffle()

            state_n.append(state)
            action_n.append(action)
            reward_n.append(reward)
            next_state_n.append(next_state)
            done_n.append(done)

        target_action_list = []
        actions_list = []
        for i in range(len(agents)):
            target_critic_action = agents[i].TargetPolicyNetwork(next_state_n[i])
            target_action = agents[i].PolicyNetwork(state_n[i])
            target_action_list.append(target_critic_action)
            actions_list.append(target_action)

        target = self.TargetQNetwork(torch.hstack(next_state_n),torch.hstack(target_action_list))
        y = reward_n[self.agent_num] + self.args.gamma*target*(1-done_n[self.agent_num])
        critic_value = self.Qnetwork(torch.hstack(state_n),torch.hstack(action_n))
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        # actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork(torch.hstack(state_n),torch.hstack(actions_list))
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            self.network_soft_updates()

    def add(self,s,action,rwd,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)

    def reset(self):

        self.replay_buffer = {agent:ReplayBuffer(self.args,agent) for agent in self.args.env_agents}
                
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents} 
        self.PolicyOptimizer = {agent:torch.optim.Adam(self.PolicyNetwork[agent].parameters(),lr=self.args.actor_lr) for agent in self.args.env_agents}
        self.TargetPolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents} 

        self.Qnetwork = AttentionCritic(self.args) 
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = AttentionCritic(self.args) 

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

        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + f"/maac_weights/{agent}/QWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + f"/maac_weights/{agent}/TargetQWeights.pth")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/maac_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maac_weights//{agent}/actorWeights.pth")
            torch.save(self.TargetPolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maac_weights/{agent}/TargetactorWeights.pth")

    def load(self,env):
        
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maac_weights/{agent}/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maac_weights/{agent}/TargetQWeights.pth",map_location=torch.device('cpu')))

        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maac_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))
            self.TargetPolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maac_weights/{agent}/TargetactorWeights.pth",map_location=torch.device('cpu')))
        
