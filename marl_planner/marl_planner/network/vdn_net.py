import torch
from torch import nn 
import torch.nn.functional as F

class VDNMixer(nn.Module):

    def __init__(self):
        super(VDNMixer,self).__init__()
                            
    
    def forward(self,q_values):

        return sum(q_values)
    
class VDNCritic(nn.Module):

    def __init__(self, args, agent):
        super(VDNCritic,self).__init__()

        self.args = args
        agent = agent
        self.input_dim = self.args.input_shape[agent]
        self.n_action = args.n_actions[agent]

        self.criticNet = nn.Sequential(
                                        nn.Linear(self.input_dim,args.critic_hidden),
                                        nn.ReLU()
                                    )
        
        self.VNet = nn.Sequential(
                                nn.Linear(args.critic_hidden,args.critic_hidden),
                                nn.ReLU(),
                                nn.Linear(args.critic_hidden,1)
                            )
        self.AdvNet = nn.Sequential(
                                    nn.Linear(args.critic_hidden,args.critic_hidden),
                                    nn.ReLU(),
                                    nn.Linear(args.critic_hidden,self.n_action)
                                )

    def forward(self,obs):

        out = self.criticNet(obs)

        V = self.VNet(out)
        Adv = self.AdvNet(out)
        Adv = Adv - Adv.mean(dim=-1,keepdim = True)
        Qval = Adv + V

        return Qval
    
