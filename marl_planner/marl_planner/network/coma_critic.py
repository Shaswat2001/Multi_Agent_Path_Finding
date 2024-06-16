import torch
from torch import nn 

class ComaCritic(nn.Module):

    def __init__(self,args):
        super(ComaCritic,self).__init__()

        input_shape = 1 + list(args.input_shape.values())[0]*args.n_agents + args.n_agents
        actions = list(args.n_actions.values())[0]
        print(args.n_actions)
        self.QNet = nn.Sequential(
            nn.Linear(input_shape,args.critic_hidden),
            nn.ReLU(),
            nn.Linear(args.critic_hidden,args.critic_hidden),
            nn.ReLU(),
            nn.Linear(args.critic_hidden,actions),
        )

    def forward(self,critic_state):
        
        Qval = self.QNet(critic_state)
        return Qval