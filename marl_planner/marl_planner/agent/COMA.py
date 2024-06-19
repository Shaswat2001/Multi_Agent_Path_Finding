import numpy as np
import torch
import os
from marl_planner.common.utils import hard_update
from marl_planner.network.coma_critic import ComaCritic
from marl_planner.common.replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from marl_planner.common.utils import onehot_from_logits

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
        self.obs_shape = self.args.input_shape[self.args.env_agents[0]]
        self.action_space = self.args.n_actions[self.args.env_agents[0]]
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
            
            state = torch.Tensor(observation[agent])
            probability = self.PolicyNetwork[agent](state)

            if stage == "training" and np.random.normal() < self.epsilon:

                act = np.random.choice(self.action_space)
                action[agent] = act
            else:
                action[agent] = int(probability.argmax(dim = 0).detach().numpy())
        
        return action
    

    def get_action_prob(self,distribution):

        # distribution = 0.9*distribution + torch.ones_like(distribution)*0.1
        distribution = distribution/torch.sum(distribution,dim=-1,keepdim=True)

        return distribution

    def learn(self):
        
        state,action,reward,next_state,done = self.replay_buffer.get_episode()

        for ai in range(len(self.args.env_agents)):
            
            agent = self.args.env_agents[ai]

            reward_i = reward[:,ai].view(-1,1)
            done_i = done[:,ai].view(-1,1)
            state_i = state[:,ai*self.obs_shape:(ai+1)*self.obs_shape]
            action_i = action[:,ai].view(-1,1)
            distribution_i = self.PolicyNetwork[agent](state_i)

            batch_size = len(reward_i)
            input_critic = self.get_critic_input(ai, state_i, action_i)
            Q_target = self.TargetQNetwork(input_critic).detach()

            action_taken = action_i.type(torch.long).reshape(-1, 1)

            # distribution = self.get_action_prob(self.get_correct_distribution(distribution_n[self.args.env_agents[i]],action_taken))
            distribution = self.get_action_prob(distribution_i)
            baseline = torch.sum(distribution * Q_target, dim=1).detach()
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            advantage = Q_taken_target - baseline

            log_pi = log_softmax(torch.gather(distribution, dim=1, index=action_taken).squeeze() +  1e-10)

            actor_loss = -torch.mean(advantage * log_pi)

            self.PolicyOptimizer[agent].zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.PolicyNetwork[self.args.env_agents[i]].parameters(), 20)
            self.PolicyOptimizer[agent].step()

            # train critic

            Q = self.Qnetwork(input_critic)

            action_taken = action_i.type(torch.long).reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            r = torch.zeros(batch_size)
            for t in range(batch_size):

                if done_i[t]:
                    r[t] = reward_i[t]
                else:
                    r[t] = reward_i[t] + self.args.gamma * Q_taken_target[t + 1] 
            
            critic_loss = torch.mean((r - Q_taken) ** 2)

            self.QOptimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.Qnetwork.parameters(), 20)
            self.QOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            hard_update(self.TargetQNetwork,self.Qnetwork)
        
        self.replay_buffer.reset()

    def get_correct_distribution(self,distribution_agent,action):


        rows = torch.arange(distribution_agent.shape[0])
        cols = action.squeeze()

        distribution_agent[rows,cols] = 1.0

        return distribution_agent

    def add(self,s,action,rwd,next_state,done):

        self.replay_buffer.store(s,action,rwd,next_state,done)

    def get_critic_input(self,id,observation,action):

        batch_size = observation.shape[0]
        id = (torch.ones(batch_size)*id).view(-1,1)

        return torch.concatenate((id,observation,action),dim=-1)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args,reward_type = "ind")
        
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