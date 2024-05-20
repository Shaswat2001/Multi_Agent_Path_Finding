import numpy as np

class AuxReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=64):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.current_mem = 0
        self.state = np.zeros(shape=(mem_size,input_shape))
        self.Vtarget = np.zeros(shape=(mem_size,1))
        self.oldPolicy = np.zeros(shape=(mem_size,n_actions))
        self.batch_size = batch_size

    def store(self,state,vtar,oldPolicy):

        for i in range(state.shape[0]):
            index = self.current_mem%self.mem_size

            self.state[index] = state[i,:]
            self.Vtarget[index] = vtar[i,:]
            self.oldPolicy[index] = oldPolicy[i,:]
            self.current_mem+=1

    def shuffle(self):
        index = self.current_mem%self.mem_size

        return (self.state[:index,:],self.Vtarget[:index,:],self.oldPolicy[:index,:])

    def reset(self):
        self.state = np.zeros(shape=(self.mem_size,self.input_shape))
        self.Vtarget = np.zeros(shape=(self.mem_size,1))
        self.oldPolicy = np.zeros(shape=(self.mem_size,self.n_actions))
        self.current_mem=0
