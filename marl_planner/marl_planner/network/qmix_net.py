import torch
from torch import nn 
import torch.nn.functional as F

class QMixer(nn.Module):

    def __init__(self,args):
        super(QMixer,self).__init__()

        self.args = args

        self.state_shape = self.args.state_shape
        self.mixer_hidden = self.args.mixer_hidden
        self.n_agents = self.args.n_agents

        self.W1Net = nn.Linear(self.state_shape,self.mixer_hidden*self.n_agents)
        self.b1Net = nn.Linear(self.state_shape,self.mixer_hidden)

        self.W2Net = nn.Linear(self.state_shape,self.mixer_hidden)
        self.b2Net = nn.Sequential(
                                nn.Linear(self.state_shape,self.mixer_hidden),
                                nn.ReLU(),
                                nn.Linear(self.mixer_hidden,1)
                            )
        
    def forward(self,state,q_values):

        batch_size = q_values.shape[0]

        q_values = q_values.view(-1,1,self.n_agents)

        W1 = torch.abs(self.W1Net(state))
        b1 = self.b1Net(state)


        W1 = W1.view(-1,self.n_agents,self.mixer_hidden)
        b1 = b1.view(-1,1,self.mixer_hidden)

        hid_val = F.elu(torch.bmm(q_values,W1)+b1)

        W2 = torch.abs(self.W2Net(state))
        b2 = self.b2Net(state)


        W2 = W2.view(-1,self.mixer_hidden,1)
        b2 = b2.view(-1,1,1)

        q_total = torch.bmm(hid_val,W2)+b2
        q_total = q_total.view(batch_size,1)

        return q_total

        