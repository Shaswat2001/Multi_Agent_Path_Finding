import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable

class ContinuousMLP(nn.Module):

    def __init__(self,args,agent):
        super(ContinuousMLP,self).__init__()

        self.args = args
        input_shape = args.input_shape[agent]
        n_action = args.n_actions[agent]
        bound = args.max_action[agent]

        self.bound = torch.tensor(bound,dtype=torch.float32)
        self.actionNet =  nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_action),
            nn.Tanh()
        )

    def forward(self,state):

        action = self.actionNet(state)
        # self.bound = self.bound.reshape(action.shape)
        action = action*self.bound

        return action

class DiscreteMLP(nn.Module):

    def __init__(self,args,agent):
        super(DiscreteMLP,self).__init__()

        self.args = args
        self.actionNet =  nn.Sequential(
            nn.Linear(args.input_shape[agent],args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.n_actions[agent]),
            nn.Softmax(dim=-1)
        )

    def forward(self,state):

        action = self.actionNet(state)

        return action
    
class SACDiscreteMLP(nn.Module):

    def __init__(self, args,agent):
        super(SACDiscreteMLP,self).__init__()

        self.args = args
        self.actionNet =  nn.Sequential(
            nn.Linear(args.input_shape[agent],args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.n_actions[agent]),
            nn.Softmax(dim=-1)
        )

    def forward(self,state):

        out = self.actionNet(state)
        int_action = torch.multinomial(out,1)
        action = Variable(torch.FloatTensor(*out.shape).fill_(0)).scatter_(1,int_action,1)
        log_prob  = F.log_softmax(out)

        return int_action,action,log_prob

class RNN(nn.Module):

    def __init__(self,args):
        super(RNN,self).__init__()

        self.args = args
        self.actionNet =  nn.Sequential(
            nn.Linear(args.input_shape,args.rnn_hidden),
            nn.ReLU()
        )
        self.rnnNet = nn.GRUCell(args.rnn_hidden,args.rnn_hidden)

        self.qNet = nn.Linear(args.rnn_hidden,args.n_actions)

    def forward(self,obs,hidden_unit):

        x = self.actionNet(obs)
        h_in = hidden_unit.reshape(-1,self.args.rnn_hidden)
        h = self.rnnNet(x,h_in)
        q = self.qNet(h)

        return q,h