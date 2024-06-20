import torch
from torch import nn 
import torch.nn.functional as F

class FOPVNetwork(nn.Module):

    def __init__(self,args,agent = None,type = None):
        super(FOPVNetwork,self).__init__()

        self.args = args
        n_agents = args.n_agents

        if type is not None:
            agent = list(args.input_shape.keys())[0]
            input_shape = args.input_shape[agent]*n_agents
        else:
            input_shape = args.input_shape[agent]

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,args.vnet_hidden),
            nn.ReLU(),
            nn.Linear(args.vnet_hidden,args.vnet_hidden//4),
            nn.ReLU(),
            nn.Linear(args.vnet_hidden//4,1)
        )

    def forward(self,state):

        vVal = self.stateNet(state)
        return vVal
    
class FOPQNetwork(nn.Module):

    def __init__(self,args,agent):
        super(FOPQNetwork,self).__init__()

        self.args = args
        input_shape = args.input_shape[agent]
        n_agents = args.n_agents
        n_action = args.n_actions[agent]

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
    
class FOPWeightedNetwork(nn.Module):

    def __init__(self,args):
        super(FOPWeightedNetwork,self).__init__()

        self.args = args
        agent = list(args.input_shape.keys())[0]
        input_shape = args.input_shape[agent]
        n_agents = args.n_agents
        n_action = args.n_actions[agent]

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape*n_agents,128),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_action*n_agents,128),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,state,action):
        
        stateProcess = self.stateNet(state)
        actionProcess = self.actionNet(action)
        ipt = torch.cat((stateProcess,actionProcess),dim=1)
        Qval = self.QNet(ipt)
        return Qval