import numpy as np

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
        return (self.state[index],self.action[index],self.reward[index],self.next_state[index],self.done[index])

class VisionReplayBuffer:

    def __init__(self,input_shape,mem_size,n_actions,rgb_size = [12,64,64],depth_size = [4,64,64],batch_size=64):
        self.mem_size = mem_size
        self.current_mem = 0
        self.statePRP = np.zeros(shape=(mem_size,input_shape))
        self.stateRGB = np.zeros(shape=[mem_size]+rgb_size)
        self.stateDepth = np.zeros(shape=[mem_size]+depth_size)
        self.action = np.zeros(shape=(mem_size,n_actions))
        self.reward = np.zeros(shape=(mem_size,1))
        self.next_statePRP = np.zeros(shape=(mem_size,input_shape))
        self.next_stateRGB = np.zeros(shape=[mem_size]+rgb_size)
        self.next_stateDepth = np.zeros(shape=[mem_size]+depth_size)
        self.done = np.zeros(shape=(mem_size,1))
        self.batch_size = batch_size

    def store(self,state,action,reward,next_state,done):
        index = self.current_mem%self.mem_size
        prp,rgb,depth = state
        next_prp,next_rgb,next_depth = next_state
        self.statePRP[index] = prp
        self.stateRGB[index] = rgb
        self.stateDepth[index] = depth
        self.action[index] = action
        self.reward[index] = reward
        self.next_statePRP[index] = next_prp
        self.next_stateRGB[index] = next_rgb
        self.next_stateDepth[index] = next_depth
        self.done[index] = done
        self.current_mem+=1

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)
        state = (self.statePRP[index],self.stateRGB[index],self.stateDepth[index])
        next_state = (self.next_statePRP[index],self.next_stateRGB[index],self.next_stateDepth[index])
        return (state,self.action[index],self.reward[index],next_state,self.done[index])