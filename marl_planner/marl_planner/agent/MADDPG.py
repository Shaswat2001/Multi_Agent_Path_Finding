import numpy as np
import torch
import os
from marl_planner.pytorch_utils import hard_update,soft_update

class MADDPG:
    '''
    MADDPG Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replayBuff = replayBuff
        self.exploration = exploration
        self.policy = policy
        self.critic = critic

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

        return action,None

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        for agent in self.args.env_agents:
            state_n = {}
            action_n = {}
            reward_n = {}
            next_state_n = {}
            done_n = {}
            for agt in self.args.env_agents:
                state,action,reward,next_state,done = self.replay_buffer[agt].shuffle()

                state_n[agt] = state
                action_n[agt] = action
                reward_n[agt] = reward
                next_state_n[agt] = next_state
                done_n[agt] = done

            target_action_list = []
            actions_list = []
            for agt in self.args.env_agents:
                target_critic_action = self.TargetPolicyNetwork[agt](next_state_n[agt])
                target_action = self.PolicyNetwork[agt](state_n[agt])
                target_action_list.append(target_critic_action)
                actions_list.append(target_action)

            target = self.TargetQNetwork[agent](torch.hstack(list(next_state_n.values())),torch.hstack(target_action_list))
            y = reward_n[agent] + self.args.gamma*target*(1-done_n[agent])
            critic_value = self.Qnetwork[agent](torch.hstack(list(state_n.values())),torch.hstack(list(action_n.values())))
            critic_loss = torch.mean(torch.square(y.detach() - critic_value),dim=1)
            self.QOptimizer[agent].zero_grad()
            critic_loss.mean().backward()
            self.QOptimizer[agent].step()

            # actions = self.PolicyNetwork(state)
            critic_value = self.Qnetwork[agent](torch.hstack(list(state_n.values())),torch.hstack(actions_list))
            actor_loss = -critic_value.mean()
            self.PolicyOptimizer[agent].zero_grad()
            actor_loss.mean().backward()
            self.PolicyOptimizer[agent].step()

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

    def add(self,s,action,rwd,next_state,done):

        for agent in self.args.env_agents:
            self.replay_buffer[agent].store(s[agent],action[0][agent],rwd[agent],next_state[agent],done[agent])

    def reset(self):

        self.replay_buffer = {agent:self.replayBuff(input_shape = self.args.state_size[agent],mem_size = self.args.mem_size,n_actions = self.args.n_actions[agent],batch_size = self.args.batch_size) for agent in self.args.env_agents}
        # Exploration Technique
        self.noiseOBJ = {agent:self.exploration(mean=np.zeros(self.args.n_actions[agent]), std_deviation=float(0.04) * np.ones(self.args.n_actions[agent])) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args.input_shape[agent],self.args.n_actions[agent],self.args.max_action[agent]) for agent in self.args.env_agents}
        self.PolicyOptimizer = {agent:torch.optim.Adam(self.PolicyNetwork[agent].parameters(),lr=self.args.actor_lr) for agent in self.args.env_agents}
        self.TargetPolicyNetwork = {agent:self.policy(self.args.input_shape[agent],self.args.n_actions[agent],self.args.max_action[agent]) for agent in self.args.env_agents}

        self.Qnetwork = {agent:self.critic(self.args.input_shape[agent],self.args.n_actions[agent],self.args.n_agents) for agent in self.args.env_agents}
        self.QOptimizer = {agent:torch.optim.Adam(self.Qnetwork[agent].parameters(),lr=self.args.critic_lr) for agent in self.args.env_agents}
        self.TargetQNetwork = {agent:self.critic(self.args.input_shape[agent],self.args.n_actions[agent],self.args.n_agents) for agent in self.args.env_agents}

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)

    def return_actions(self,state):

        return self.PolicyNetwork(state),self.TargetPolicyNetwork(state)
    
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