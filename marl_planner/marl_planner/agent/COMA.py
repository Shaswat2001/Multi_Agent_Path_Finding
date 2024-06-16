import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update
from marl_planner.network.coma_critic import ComaCritic
from marl_planner.common.replay_buffer import ReplayBuffer
from torch.distributions import Categorical

class COMA:
    '''
    COMA Algorithm 
    '''
    def __init__(self,args,policy):

        self.args = args # Argument values given by the user
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min

        self.epsilon_decay = (self.epsilon - self.epsilon_min)/50000
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.policy = policy

        self.reset()

    def choose_action(self,observation,stage="training"):

        self.learning_step+=1

        if self.learning_step < 50000:
            self.epsilon -= self.epsilon_decay

            self.epsilon = max(self.epsilon,self.epsilon_min)
        else:
            self.epsilon = self.epsilon_min

        action = {}
        for agent in self.args.env_agents:
            
            
            if np.random.random() < self.epsilon and stage == "training":
                action[agent] = np.random.choice(self.args.n_actions[agent])
            else:
                state = torch.Tensor(observation[agent])
                distribution = self.PolicyNetwork[agent](state)
                act_n = distribution.max(dim=0)[1].detach().cpu().numpy()
                action[agent] = act_n

        return action
    

    def get_action_prob(self,distribution):

        # distribution = 0.9*distribution + torch.ones_like(distribution)*0.1
        distribution = distribution/torch.sum(distribution,dim=-1,keepdim=True)

        return distribution

    def learn(self):
        
        state_n = {}
        action_n = {}
        reward_n = {}
        distribution_n = {}
        next_state_n = {}
        done_n = {}

        for agt in self.args.env_agents:

            state,action,reward,next_state,done = self.replay_buffer[agt].get_episode()

            state_n[agt] = state
            action_n[agt] = action
            reward_n[agt] = reward
            distribution_n[agt] = self.get_action_prob(self.PolicyNetwork[agt](state))
            next_state_n[agt] = next_state
            done_n[agt] = done

        for i in range(len(self.args.env_agents)):
            
            batch_size = len(reward_n[self.args.env_agents[i]])
            input_critic = self.get_critic_input(i, state_n, action_n)
            Q_target = self.TargetQNetwork(input_critic).detach()

            action_taken = action_n[self.args.env_agents[i]].type(torch.long).reshape(-1, 1)

            baseline = torch.sum(distribution_n[self.args.env_agents[i]] * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze().detach()
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(distribution_n[self.args.env_agents[i]], dim=1, index=action_taken).squeeze() + 1e-10)

            actor_loss = -torch.mean(advantage * log_pi)

            self.PolicyOptimizer[self.args.env_agents[i]].zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.PolicyNetwork[self.args.env_agents[i]].parameters(), 20)
            self.PolicyOptimizer[self.args.env_agents[i]].step()

            # train critic

            Q = self.Qnetwork(input_critic)

            action_taken = action_n[self.args.env_agents[i]].type(torch.long).reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            r = torch.zeros(batch_size)
            for t in range(batch_size):

                if done_n[self.args.env_agents[i]][t]:
                    r[t] = reward_n[self.args.env_agents[i]][t]
                else:
                    r[t] = reward_n[self.args.env_agents[i]][t] + self.args.gamma * Q_taken_target[t + 1] * (1 - done_n[self.args.env_agents[i]][t])
            
            critic_loss = torch.mean((r - Q_taken) ** 2)

            self.QOptimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Qnetwork.parameters(), 20)
            self.QOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            hard_update(self.TargetQNetwork,self.Qnetwork)
        
        for agt in self.args.env_agents:
            self.replay_buffer[agt].reset()

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

        self.replay_buffer = {agent:ReplayBuffer(self.args,agent) for agent in self.args.env_agents}
        
        self.PolicyNetwork = {agent:self.policy(self.args,agent) for agent in self.args.env_agents}
        self.PolicyOptimizer = {agent:torch.optim.Adam(self.PolicyNetwork[agent].parameters(),lr=self.args.actor_lr) for agent in self.args.env_agents}

        self.Qnetwork = ComaCritic(self.args)
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = ComaCritic(self.args)

        hard_update(self.TargetQNetwork,self.Qnetwork)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        for agent in self.args.env_agents:
            os.makedirs("config/saves/training_weights/"+ env + f"/coma_weights/{agent}", exist_ok=True)
            torch.save(self.PolicyNetwork[agent].state_dict(),"config/saves/training_weights/"+ env + f"/coma_weights//{agent}/actorWeights.pth")

    def load(self,env):

        for agent in self.args.env_agents:
            self.PolicyNetwork[agent].load_state_dict(torch.load("config/saves/training_weights/"+ env + f"/coma_weights//{agent}/actorWeights.pth",map_location=torch.device('cpu')))