import numpy as np
import torch
import os
from marl_planner.pytorch_utils import hard_update,soft_update
from torch.distributions import Categorical

class COMA:
    '''
    COMA Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replayBuff = replayBuff
        self.policy = policy
        self.critic = critic

        self.reset()

    def choose_action(self,observation,stage="training"):

        action = {}
        dist_n = {}
        for agent in self.args.env_agents:

            state = torch.Tensor(observation[agent])
            distribution = self.PolicyNetwork[agent](state)
            act_n = Categorical(distribution).sample()
        
            action[agent] = act_n
            dist_n[agent] = distribution

        return (action, dist_n)

    def learn(self):
        
        self.learning_step+=1

        if self.learning_step<self.args.batch_size:
            return
        
        state_n = {}
        action_n = {}
        reward_n = {}
        distribution_n = {}
        next_state_n = {}
        done_n = {}
        for agt in self.args.env_agents:
            state,action,reward,distribution,next_state,done = self.replay_buffer[agt].shuffle()

            state_n[agt] = torch.vstack(state)
            action_n[agt] = torch.vstack(action)
            reward_n[agt] = torch.vstack(reward)
            distribution_n[agt] = torch.vstack(distribution)
            next_state_n[agt] = torch.vstack(next_state)
            done_n[agt] = torch.vstack(done)

        for i in range(len(self.args.env_agents)):

            input_critic = self.get_critic_input(i, state_n, action_n)
            Q_target = self.TargetQNetwork(input_critic).detach()

            action_taken = action_n[self.args.env_agents[i]].type(torch.long).reshape(-1, 1)

            baseline = torch.sum(distribution_n[self.args.env_agents[i]] * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(distribution_n[self.args.env_agents[i]], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            self.PolicyOptimizer[self.args.env_agents[i]].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.PolicyNetwork[self.args.env_agents[i]].parameters(), 5)
            self.PolicyOptimizer[self.args.env_agents[i]].step()

            # train critic

            Q = self.Qnetwork(input_critic)

            action_taken = action_n[self.args.env_agents[i]].type(torch.long).reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            r = torch.zeros(len(reward_n[self.args.env_agents[i]]))
            for t in range(len(reward_n[self.args.env_agents[i]])-1):
                if done_n[self.args.env_agents[i]][t]:
                    r[t] = reward_n[self.args.env_agents[i]][t]
                else:
                    r[t] = reward_n[self.args.env_agents[i]][t] + self.args.gamma * Q_taken_target[t + 1]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            self.QOptimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.Qnetwork.parameters(), 5)
            self.QOptimizer.step()


        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

        
        for agt in self.args.env_agents:
            self.replay_buffer[agt].reset()

    def add(self,s,action,rwd,next_state,done):

        for agent in self.args.env_agents:
            self.replay_buffer[agent].store(s[agent],action[0][agent],rwd[agent],action[1][agent],next_state[agent],done[agent])

    def get_critic_input(self,id,observation,action):

        id = (torch.ones(self.args.batch_size)*id).view(-1,1)
        observations = torch.hstack(list(observation.values()))
        action = torch.hstack(list(action.values()))

        return torch.concatenate((id,observations,action),dim=-1)

    def reset(self):

        self.replay_buffer = {agent:self.replayBuff(input_shape = self.args.state_size[agent],mem_size = self.args.mem_size,n_actions = self.args.n_actions[agent],batch_size = self.args.batch_size) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args.input_shape[agent],self.args.n_actions[agent]) for agent in self.args.env_agents}
        self.PolicyOptimizer = {agent:torch.optim.Adam(self.PolicyNetwork[agent].parameters(),lr=self.args.actor_lr) for agent in self.args.env_agents}

        self.Qnetwork = self.critic(list(self.args.input_shape.values())[0],list(self.args.n_actions.values())[0],self.args.n_agents)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = self.critic(list(self.args.input_shape.values())[0],list(self.args.n_actions.values())[0],self.args.n_agents)

        hard_update(self.TargetQNetwork,self.Qnetwork)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/maddpg_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/maddpg_weights//{agent}/actorWeights.pth")

    def load(self,env):

        for agent in self.args.env_agents:
            self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/maddpg_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))