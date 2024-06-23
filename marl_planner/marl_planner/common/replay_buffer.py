import numpy as np
import torch

class ReplayBuffer:

    def __init__(self,args,reward_type = "global",action_space = "discrete"):
        self.args = args
        self.mem_size = args.mem_size
        self.env_agents = args.env_agents
        self.reward_type = reward_type
        self.action_space = action_space
        self.current_mem = 0
        self.observation = [0]*self.mem_size
        self.state = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.next_observation = [0]*self.mem_size
        self.next_state = [0]*self.mem_size
        self.done = [0]*self.mem_size
        self.batch_size = args.batch_size

    def store(self,state,observation,action,reward,next_state,next_observation,done):

        if self.current_mem < self.mem_size:
            
            index = self.current_mem
        else:
            index = self.current_mem%self.mem_size

        self.state[index] = state
        self.observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.next_observation[index] = next_observation
        self.done[index] = done
        self.current_mem+=1

    def reset(self):

        self.current_mem = 0
        self.observation = [0]*self.mem_size
        self.state = [0]*self.mem_size
        self.action = [0]*self.mem_size
        self.reward = [0]*self.mem_size
        self.next_observation = [0]*self.mem_size
        self.next_state = [0]*self.mem_size
        self.done = [0]*self.mem_size

    def get_episode(self):

        index = self.current_mem%self.mem_size
        state = []
        observation = []
        action = []
        reward = []
        next_state = []
        next_observation = []
        done = []
        for idx in range(index):

            state.append(torch.Tensor(self.state[idx]))
            observation.append(torch.hstack([torch.Tensor(obs) for obs in self.observation[idx].values()]))

            if self.action_space == "discrete":
                action.append(torch.hstack([torch.Tensor([act]).to(torch.int64) for act in self.action[idx].values()]))
            else:
                action.append(torch.hstack([torch.Tensor(act) for act in self.action[idx].values()]))

            if self.reward_type == "global":
                reward.append(torch.Tensor([self.reward[idx]]))
            else:
                print(self.reward[idx])
                reward.append(torch.hstack([torch.Tensor([rwd]) for rwd in self.reward[idx].values()]))

            next_state.append(torch.Tensor(self.next_state[idx]))
            next_observation.append(torch.hstack([torch.Tensor(nxt_obs) for nxt_obs in self.next_observation[idx].values()]))
            done.append(torch.hstack([torch.Tensor([dn]) for dn in self.done[idx].values()]))
        
        state = torch.vstack(state)
        observation = torch.vstack(observation)
        action = torch.vstack(action)
        reward = torch.vstack(reward)
        next_state = torch.vstack(next_state)
        next_observation = torch.vstack(next_observation)
        done = torch.vstack(done)

        return (state,observation,action,reward,next_state,next_observation,done)

    def shuffle(self):
        max_mem = min(self.mem_size, self.current_mem)
        index = np.random.choice(max_mem, self.batch_size)

        state = []
        observation = []
        action = []
        reward = []
        next_observation = []
        next_state = []
        done = []
        for idx in index:
            state.append(torch.Tensor(self.state[idx]))
            observation.append(torch.hstack([torch.Tensor(obs) for obs in self.observation[idx].values()]))

            if self.action_space == "discrete":
                action.append(torch.hstack([torch.Tensor([act]).to(torch.int64) for act in self.action[idx].values()]))
            else:
                action.append(torch.hstack([torch.Tensor(act) for act in self.action[idx].values()]))

            if self.reward_type == "global":
                reward.append(torch.Tensor([self.reward[idx]]))
            else:
                reward.append(torch.hstack([torch.Tensor([rwd]) for rwd in self.reward[idx].values()]))

            next_state.append(torch.Tensor(self.next_state[idx]))
            next_observation.append(torch.hstack([torch.Tensor(nxt_obs) for nxt_obs in self.next_observation[idx].values()]))
            done.append(torch.hstack([torch.Tensor([dn]) for dn in self.done[idx].values()]))
        
        state = torch.vstack(state)
        observation = torch.vstack(observation)
        action = torch.vstack(action)
        reward = torch.vstack(reward)
        next_state = torch.vstack(next_state)
        next_observation = torch.vstack(next_observation)
        done = torch.vstack(done)

        return (state,observation,action,reward,next_state,next_observation,done)
    
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
    