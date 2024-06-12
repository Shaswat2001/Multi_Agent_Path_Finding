from os import stat
import torch
from torch import nn 
from marl_planner.pytorch_utils import *
from torch.nn.init import uniform_

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.01)

class PolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action,bound):
        super(PolicyNetwork,self).__init__()


        self.bound = torch.tensor(bound,dtype=torch.float32)
        self.actionNet =  nn.Sequential(
            nn.Linear(input_shape,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_action),
            nn.Tanh()
        )

    def forward(self,state):

        action = self.actionNet(state)
        # self.bound = self.bound.reshape(action.shape)
        action = action*self.bound

        return action
    
class DiscretePolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(DiscretePolicyNetwork,self).__init__()

        self.actionNet =  nn.Sequential(
            nn.Linear(input_shape,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_action),
            nn.Softmax(dim=-1)
        )

    def forward(self,state):

        action = self.actionNet(state)

        return action

class CentralizedQNetwork(nn.Module):

    def __init__(self,input_shape,n_action,n_agents):
        super(CentralizedQNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape*n_agents,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_action*n_agents,128),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,state,action):
        
        stateProcess = self.stateNet(state)
        actionProcess = self.actionNet(action)
        ipt = torch.cat((stateProcess,actionProcess),dim=1)
        Qval = self.QNet(ipt)
        return Qval

class QNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(QNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_action,128),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    
    def forward(self,state,action):
        
        stateProcess = self.stateNet(state)
        actionProcess = self.actionNet(action)
        ipt = torch.cat((stateProcess,actionProcess),dim=1)
        Qval = self.QNet(ipt)
        return Qval
    
class DiscreteQNetwork(nn.Module):

    def __init__(self,input_shape,n_action,n_agents):
        super(DiscreteQNetwork,self).__init__()

        self.input_dim = 1 + input_shape*n_agents + n_agents

        self.QNet = nn.Sequential(
            nn.Linear(self.input_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_action),
        )

    def forward(self,critic_state):
        
        Qval = self.QNet(critic_state)
        return Qval

class VNetwork(nn.Module):

    def __init__(self,input_shape):
        super(VNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,state):

        vVal = self.stateNet(state)
        return vVal