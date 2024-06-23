import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable
from marl_planner.common.utils import *
    
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
            nn.Sigmoid()
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
    
class DiscreteGaussianNet(nn.Module):

    def __init__(self, args,agent):
        super(DiscreteGaussianNet,self).__init__()

        self.args = args
        self.actionNet =  nn.Sequential(
            nn.Linear(args.input_shape[agent],args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.policy_hidden),
            nn.ReLU(),
            nn.Linear(args.policy_hidden,args.n_actions[agent]),
        )

    def forward(self,state,sample = False):

        out = self.actionNet(state)
        probability = F.softmax(out,dim = 1)

        if sample:
            int_action,action = categorical_sample(probability)
        else:
            action = onehot_from_logits(probability)

        log_prob  = F.log_softmax(out)

        if sample:
            specific_log_prob = log_prob.gather(1,int_action)
        else:
            specific_log_prob = None

        regularise_action = (out**2).mean()

        entropy = -(log_prob*probability).sum(1).mean()

        output = PolicyOps(action=action, log_prob=log_prob, entropy=entropy, pi=out, reg_pi=regularise_action, specific_log_prob=specific_log_prob)

        return output
    
class ContGaussianNet(nn.Module):

    def __init__(self, args,agent):
        super(ContGaussianNet,self).__init__()

        self.args = args
        self.input_shape = args.input_shape[agent]
        self.n_actions = args.n_actions[agent]
        self.log_std_min = args.log_std_min[agent]
        self.log_std_max = args.log_std_max[agent]
        self.act_lim = torch.tensor(args.bound,dtype=torch.float32)

        self.actorNet = nn.Sequential(
            nn.Linear(self.n_actions,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
        )

        self.meanNet = nn.Sequential(
            nn.Linear(256,self.n_actions)
        )

        self.stdNet = nn.Sequential(
            nn.Linear(256,self.n_actions)
        )
    
    def forward(self,state):

        X = self.actorNet(state)
        mean = self.meanNet(X)
        log_std = self.stdNet(X)
        log_std = torch.tanh(log_std.detach())
        log_std = Squash(in_min=-1, in_max=1, out_min=self.log_std_min, out_max=self.log_std_max)(log_std)
        pi = DiagonalGaussianSample()(mean=mean, log_std=log_std)
        tanh_pi = torch.tanh(pi)
        log_prob_tanh_pi = TanhDiagonalGaussianLogProb()(gaussian_samples=pi,
                                                            tanh_gaussian_samples=tanh_pi,
                                                            mean=mean,
                                                            log_std=log_std)
        tanh_mean = torch.tanh(mean)
        scaled_tanh_pi = tanh_pi * self.act_lim
        scaled_tanh_mean = tanh_mean * self.act_lim

        output = PolicyOpsCont(raw_mean=mean, mean=scaled_tanh_mean,log_std=log_std, pi=scaled_tanh_pi, log_prob_pi=log_prob_tanh_pi)

        return output


class RNN(nn.Module):

    def __init__(self,args,agent):
        super(RNN,self).__init__()

        self.args = args
        self.input_shape = args.input_shape[agent]
        self.actionNet =  nn.Sequential(
            nn.Linear(self.input_shape,args.rnn_hidden),
            nn.ReLU()
        )
        self.rnnNet = nn.GRUCell(args.rnn_hidden,args.rnn_hidden)

        self.qNet = nn.Linear(args.rnn_hidden,args.n_actions[agent])

    def forward(self,obs,hidden_unit):
        
        obs = obs.reshape(-1,self.input_shape)
        x = self.actionNet(obs)
        h_in = hidden_unit.reshape(-1,self.args.rnn_hidden)
        h = self.rnnNet(x,h_in)
        q = self.qNet(h)

        return q,h