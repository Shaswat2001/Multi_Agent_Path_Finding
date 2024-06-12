import numpy as np
import torch
import os
from marl_planner.pytorch_utils import hard_update,soft_update

class MAAC:
    '''
    MAAC Algorithm 
    '''
    def __init__(self,args,policy,critic,replayBuff,exploration,agent_num):

        self.args = args # Argument values given by the user
        self.learning_step = 0 # counter to keep track of learning
        # Replay Buffer provided by the user
        self.replayBuff = replayBuff
        self.exploration = exploration
        self.agent_num = agent_num
        self.policy = policy
        self.critic = critic

        self.reset()

    def choose_action(self,state,stage="training"):

        state = torch.Tensor(state)
        
        if stage == "training":
            action = self.PolicyNetwork(state).detach().numpy()
            action += self.noiseOBJ()
        else:
            action = self.TargetPolicyNetwork(state).detach().numpy()

        action = np.clip(action,self.args.min_action,self.args.max_action)

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
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.Qnetwork,self.args.tau)

    def add(self,s,action,rwd,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)

    def reset(self):

        self.replay_buffer = self.replayBuff(input_shape = self.args.state_size,mem_size = self.args.mem_size,n_actions = self.args.n_actions,batch_size = self.args.batch_size)
        # Exploration Technique
        self.noiseOBJ = self.exploration(mean=np.zeros(self.args.n_actions), std_deviation=float(0.04) * np.ones(self.args.n_actions))
        
        self.PolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=self.args.actor_lr)
        self.TargetPolicyNetwork = self.policy(self.args.input_shape,self.args.n_actions,self.args.max_action)

        self.Qnetwork = self.critic(self.args.input_shape,self.args.n_actions,self.args.n_agents) 
        self.QOptimizer = torch.optim.Adam(self.Qnetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = self.critic(self.args.input_shape,self.args.n_actions,self.args.n_agents) 

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.Qnetwork)

    def return_actions(self,state):

        return self.PolicyNetwork(state),self.TargetPolicyNetwork(state)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/maddpg_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/maddpg_weights/actorWeights.pth")
        torch.save(self.Qnetwork.state_dict(),"config/saves/training_weights/"+ env + "/maddpg_weights/QWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/maddpg_weights/TargetactorWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + "/maddpg_weights/TargetQWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/maddpg_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.Qnetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/maddpg_weights/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/maddpg_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/maddpg_weights/TargetQWeights.pth",map_location=torch.device('cpu')))