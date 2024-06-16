import numpy as np
import torch

class ReplayBuffer:

    def __init__(self,args,agent):
        self.args = args
        self.agent = agent
        self.mem_size = args.mem_size
        self.current_mem = 0
        self.observation = np.zeros(shape=(args.mem_size,args.input_shape[agent]))
        self.action = np.zeros(shape=(args.mem_size,args.action_space[agent]))
        self.reward = np.zeros(shape=(args.mem_size,1))
        self.next_observation = np.zeros(shape=(args.mem_size,args.input_shape[agent]))
        self.done = np.zeros(shape=(args.mem_size,1))
        self.batch_size = args.batch_size

    def store(self,observation,action,reward,next_observation,done):
        index = self.current_mem%self.mem_size
        self.observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.next_observation[index] = next_observation
        self.done[index] = done
        self.current_mem+=1

    def reset(self):

        self.current_mem = 0
        self.observation = np.zeros(shape=(self.mem_size,self.args.input_shape[self.agent]))
        self.action = np.zeros(shape=(self.mem_size,self.args.action_space[self.agent]))
        self.reward = np.zeros(shape=(self.mem_size,1))
        self.next_observation = np.zeros(shape=(self.mem_size,self.args.input_shape[self.agent]))
        self.done = np.zeros(shape=(self.mem_size,1))

    def get_episode(self):

        index = self.current_mem%self.mem_size
        observation = torch.Tensor(self.observation[:index])
        action = torch.Tensor(self.action[:index])
        reward = torch.Tensor(self.reward[:index])
        next_observation = torch.Tensor(self.next_observation[:index])
        done = torch.Tensor(self.done[:index])

        return (observation,action,reward,next_observation,done)

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)

        observation = torch.Tensor(self.observation[index])
        action = torch.Tensor(self.action[index])
        reward = torch.Tensor(self.reward[index])
        next_observation = torch.Tensor(self.next_observation[index])
        done = torch.Tensor(self.done[index])

        return (observation,action,reward,next_observation,done)
    
class ReplayBufferEpisode:

    def __init__(self,args,agent):
        self.args = args
        self.agent = agent
        self.mem_size = args.mem_size
        self.current_mem = 0
        self.observation = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.next_observation = [0]*self.mem_size
        self.done = [0]*self.mem_size
        self.batch_size = args.batch_size

    def store(self,observation,action,reward,next_observation,done):
        index = self.current_mem%self.mem_size
        self.observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.next_observation[index] = next_observation
        self.done[index] = done
        self.current_mem+=1

    def reset(self):

        self.current_mem = 0
        self.observation = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.next_observation = [0]*self.mem_size
        self.done = [0]*self.mem_size

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)

        observation = []
        action = []
        reward = []
        next_observation = []
        done = []
        for idx in index:
            observation.append(torch.Tensor(self.observation[idx]))
            action.append(torch.Tensor(self.action[idx]))
            reward.append(torch.Tensor(self.reward[idx]))
            next_observation.append(torch.Tensor(self.next_observation[idx]))
            done.append(torch.Tensor(self.done[idx]))

        return (observation,action,reward,next_observation,done)
    