import numpy as np

class ConstReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=256):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.current_mem = 0
        self.state = np.zeros(shape=(mem_size,input_shape))
        self.newConstraint = np.zeros(shape=(mem_size,2*n_actions))
        self.oldConstraint = np.zeros(shape=(mem_size,2*n_actions))
        self.action = np.zeros(shape=(mem_size,n_actions))
        self.batch_size = batch_size

    def store(self,state,newconst,oldconst,action):

        index = self.current_mem%self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.newConstraint[index] = newconst
        self.oldConstraint[index] = oldconst
        self.current_mem+=1

    def shuffle(self):

        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)
        return (self.state[index],self.action[index],self.newConstraint[index],self.oldConstraint[index])

    def reset(self):
        self.state = np.zeros(shape=(self.mem_size,self.input_shape))
        self.newConstraint = np.zeros(shape=(self.mem_size,2*self.n_actions))
        self.oldConstraint = np.zeros(shape=(self.mem_size,2*self.n_actions))
        self.action = np.zeros(shape=(self.mem_size,self.n_actions))
        self.current_mem=0

class CostReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,batch_size=64):
        self.mem_size = mem_size
        self.current_mem = 0
        self.state = np.zeros(shape=(mem_size,input_shape))
        self.action = np.zeros(shape=(mem_size,n_actions))
        self.reward = np.zeros(shape=(mem_size,1))
        self.constraint = np.zeros(shape=(mem_size,1))
        self.next_state = np.zeros(shape=(mem_size,input_shape))
        self.done = np.zeros(shape=(mem_size,1))
        self.batch_size = batch_size

    def store(self,state,action,reward,constraint,next_state,done):
        index = self.current_mem%self.mem_size
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.constraint[index] = constraint
        self.next_state[index] = next_state
        self.done[index] = done
        self.current_mem+=1

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)
        return (self.state[index],self.action[index],self.reward[index],self.constraint[index],self.next_state[index],self.done[index])
