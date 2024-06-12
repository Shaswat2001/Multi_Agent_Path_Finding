import numpy as np
import torch

class ReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=64):
        self.mem_size = mem_size
        self.current_mem = 0
        self.state = np.zeros(shape=(mem_size,input_shape))
        self.action = np.zeros(shape=(mem_size,n_actions))
        self.reward = np.zeros(shape=(mem_size,1))
        self.next_state = np.zeros(shape=(mem_size,input_shape))
        self.done = np.zeros(shape=(mem_size,1))
        self.batch_size = batch_size

    def store(self,state,action,reward,next_state,done):
        index = self.current_mem%self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.done[index] = done
        self.current_mem+=1

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)

        state = torch.Tensor(self.state[index])
        action = torch.Tensor(self.action[index])
        reward = torch.Tensor(self.reward[index])
        next_state = torch.Tensor(self.next_state[index])
        done = torch.Tensor(self.done[index])

        return (state,action,reward,next_state,done)
    
class DiscreteReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=64):
        self.mem_size = mem_size
        self.current_mem = 0
        self.state = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.pi = [0]*self.mem_size
        self.next_state = [0]*self.mem_size
        self.done = [0]*self.mem_size
        self.batch_size = batch_size

    def reset(self):
        self.current_mem = 0
        self.state = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.pi = [0]*self.mem_size
        self.next_state = [0]*self.mem_size
        self.done = [0]*self.mem_size

    def store(self,state,action,reward,pi,next_state,done):
        index = self.current_mem%self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.pi[index] = pi
        self.next_state[index] = next_state
        self.done[index] = done
        self.current_mem+=1

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)
        state = []
        action = []
        reward = []
        pi = []
        next_state = []
        done = []

        for i in index:

            state.append(torch.Tensor(self.state[i]))
            action.append(torch.Tensor(self.action[i]))
            reward.append(torch.Tensor([self.reward[i]]))
            pi.append(torch.Tensor(self.pi[i]))
            next_state.append(torch.Tensor(self.next_state[i]))
            done.append(torch.Tensor([self.done[i]]))

        return (state,action,reward,pi,next_state,done)