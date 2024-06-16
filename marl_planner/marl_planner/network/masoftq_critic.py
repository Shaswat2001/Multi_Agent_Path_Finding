import torch
from torch import nn 

class MASoftQCritic(nn.Module):

    def __init__(self,args):
        super(MASoftQCritic,self).__init__()

        self.args = args
        agent = list(args.input_shape.keys())[0]
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