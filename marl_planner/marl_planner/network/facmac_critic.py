import torch
from torch import nn 

class FACMACCritic(nn.Module):

    def __init__(self,args,agent):
        super(FACMACCritic,self).__init__()

        self.args = args
        input_shape = args.input_shape[agent]
        n_action = args.n_actions[agent]

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_action,128),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,state,action):
        
        stateProcess = self.stateNet(state)
        actionProcess = self.actionNet(action)
        ipt = torch.cat((stateProcess,actionProcess),dim=1)
        Qval = self.QNet(ipt)
        return Qval