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

class CEM(nn.Module):

    def __init__(self,n_action,init_mean = 0,init_std = 1,no_samples = 64):
        super(CEM,self).__init__()
        self.n_action = n_action
        self.mean = init_mean*np.ones(self.n_action)
        self.std = init_std*np.ones(self.n_action)
        self.no_samples = no_samples
        self.no_iterations = 10

    def forward(self,state):

        cem_state = torch.Tensor(np.vstack([state]*self.no_samples))

        for i in range(self.no_iterations):

            pass

class NAFNetwork(nn.Module):

    def __init__(self,input_shape,n_action,bound):
        super(NAFNetwork,self).__init__()

        self.bound = bound

        self.actionNet =  nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_action),
            nn.Tanh()
        )

        self.subNet =nn.Sequential(
            nn.Linear(input_shape,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )

        self.VNet = nn.Sequential(
            nn.Linear(256,1)
        )

        self.TNet = nn.Sequential(
            nn.Linear(256,int(n_action*(1+n_action)/2))
        )
    
    def forward(self,state):
        
        X = self.actionNet(state)
        action = X*self.bound

        val = self.subNet(state)
        V = self.VNet(val)
        Tmat = self.TNet(val)

        return action,V,Tmat        

class PhasicPolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_actions,log_std_min,log_std_max,act_lim):
        super(PhasicPolicyNetwork,self).__init__()

        self.n_actions = n_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_lim = act_lim

        self.actorNet = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
        )

        self.meanNet = nn.Sequential(
            nn.Linear(256,n_actions)
        )

        self.stdNet = nn.Sequential(
            nn.Tanh(256,n_actions)
        )

        self.valueNet = nn.Sequential(
            nn.Linear(256, 1)
        )
    
    def forward(self,state):

        X = self.actorNet(state)
        V = self.valueNet(X)
        mean = self.meanNet(X)
        log_std = self.stdNet(X)
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

        output = PolicyOps(raw_mean=mean, mean=scaled_tanh_mean,log_std=log_std, pi=scaled_tanh_pi, log_prob_pi=log_prob_tanh_pi)

        return output,V
    
class SafePolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_actions,bound):
        super(SafePolicyNetwork,self).__init__()

        self.n_actions = n_actions
        self.bound = torch.tensor(bound,dtype=torch.float32)

        self.actorNet = nn.Sequential(
            nn.Linear(input_shape,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_actions),
            nn.Tanh()
        )

        self.valueNet = nn.Sequential(
            nn.Linear(n_actions, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    
    def forward(self,state):

        action = self.actorNet(state)
        action = action*self.bound
        safeQ = self.valueNet(action)
    
        return action,safeQ

class GaussianPolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action,bound,log_std_min,log_std_max):
        super(GaussianPolicyNetwork,self).__init__()

        self.n_actions = n_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_lim = torch.tensor(bound,dtype=torch.float32)

        self.actorNet = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
        )

        self.meanNet = nn.Sequential(
            nn.Linear(256,n_action)
        )

        self.stdNet = nn.Sequential(
            nn.Linear(256,n_action)
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

        output = PolicyOps(raw_mean=mean, mean=scaled_tanh_mean,log_std=log_std, pi=scaled_tanh_pi, log_prob_pi=log_prob_tanh_pi)

        return output

class MeanPolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(MeanPolicyNetwork,self).__init__()

        self.meanNet = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_action)
        )
    
    def forward(self,state):

        meanVal = self.meanNet(state)
        return meanVal

class StdPolicyNetwork(nn.Module):

    def __init__(self,input_shape,n_action,log_std_min,log_std_max):
        super(StdPolicyNetwork,self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.stdNet = nn.Sequential(
            nn.Linear(input_shape,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_action),
            nn.Tanh()
        )
    
    def forward(self,state):

        stdVal = self.stdNet(state)
        stdVal = Squash(in_min=-1, in_max=1, out_min=self.log_std_min, out_max=self.log_std_max)(stdVal)
        return stdVal

class ConstraintNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(ConstraintNetwork,self).__init__()

        self.constNet = nn.Sequential(
            nn.Linear(input_shape,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,n_action),
            nn.Tanh()
        )

    def forward(self,state):
        
        corrected_action = self.constNet(state)

        return corrected_action
    
class MultiplierNetwork(nn.Module):

    def __init__(self,input_shape):
        super(MultiplierNetwork,self).__init__()

        self.constNet = nn.Sequential(
            nn.Linear(input_shape,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )

    def forward(self,state):
        
        mult_val = self.constNet(state)

        return mult_val

class PhasicQNetwork(nn.Module):

    def __init__(self,input_shape,n_actions):
        super(PhasicQNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
        )

        self.actionNet = nn.Sequential(
            nn.Linear(n_actions,32),
            nn.ReLU(),
        )

        self.QNet = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1)
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

class DiscreteActorNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(DiscreteActorNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_action),
            nn.Softmax(dim=1)
        )
    
    def forward(self,state):
        
        val = self.stateNet(state)
        return val

class DiscreteQNetwork(nn.Module):

    def __init__(self,input_shape,n_action):
        super(DiscreteQNetwork,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_action),
            nn.Identity()
        )
    
    def forward(self,state):
        
        Qval = self.stateNet(state)
        return Qval
    
class CouplingLayer(nn.Module):

    def __init__(self,n_action):
        super(CouplingLayer,self).__init__()

        self.n_action = n_action
        self.PolicyNet = nn.Sequential(
            nn.Linear(n_action//2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )

        self.PolicyScale = nn.Sequential(
            nn.Linear(64,n_action//2),
            nn.Sigmoid()
        )

        self.PolicyTrans = nn.Sequential(
            nn.Linear(64,n_action//2)
        )
    
    def forward(self,action):
        
        prediction = self.PolicyNet(action)
        log_s = self.PolicyScale(prediction)
        trans = self.PolicyTrans(prediction)

        return log_s,trans
    
class RealNVP(nn.Module):

    def __init__(self,n_action):
        super(RealNVP,self).__init__()

        self.n_action = n_action
        self.PolicyCPL = CouplingLayer(n_action)
        self.SafePolicyCPL = CouplingLayer(n_action)
    
    def forward(self,action,safe_action):
        
        act1,act2 = self.split(action)
        log_s,trans = self.PolicyCPL(act1)
        sfprd1 = act1
        sfprd2 = act2 * torch.exp(log_s) + trans
        sfprd = torch.concatenate([sfprd1, sfprd2], axis=-1)

        sact1,sact2 = self.split(safe_action)
        log_s,trans = self.SafePolicyCPL(sact1)
        prd1 = sact1
        prd2 = (sact2 - trans)/torch.exp(log_s)
        prd = torch.concatenate([prd1, prd2], axis=-1)

        return sfprd,prd
    
    def split(self, x):
        dim = self.n_action
        x = torch.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]

class DQN(nn.Module):

    def __init__(self,input_shape,n_action):
        super(DQN,self).__init__()

        self.stateNet = nn.Sequential(
            nn.Linear(input_shape,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,n_action)
        )
    
    def forward(self,state):
        
        Qval = self.stateNet(state)
        return Qval
    
class FeatureExtractor(nn.Module):

    def __init__(self,depth_input,rgb_input,prpt_input):

        super(FeatureExtractor,self).__init__()

        self.depthConv = nn.Sequential(
                            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2),
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=4,stride=4),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2),
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Flatten()
                        )
        
        self.rgbConv = nn.Sequential(
                        nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2,stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=4,stride=4),
                        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Flatten()
                    )


        self.prptNet = nn.Sequential(
            nn.Linear(prpt_input,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )

        self.transEncoder = TransformerEncoder(2,4352)

        self.projectHead = nn.Sequential(
            nn.Linear(4352,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )

    def forward(self,vision_dpt,vision_rgb,proprioceptive):

        depth_process = self.depthConv(vision_dpt)
        rgb_process = self.rgbConv(vision_rgb)
        depth_process = depth_process.view(depth_process.shape[0],-1)
        rgb_process = rgb_process.view(rgb_process.shape[0],-1)

        pprt_process = self.prptNet(proprioceptive)
        data = torch.cat((pprt_process,rgb_process,depth_process),dim=-1)

        transform_encd = self.transEncoder(data)
        observation = self.projectHead(transform_encd)

        return observation

class TransformerEncoder(nn.Module):

    def __init__(self,n_hidden,input_shape):
        super(TransformerEncoder,self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_shape, nhead=n_hidden)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self,observation):

        return self.transformer_encoder(observation)


if __name__ == "__main__":

    rgb = torch.randn(2,12,64,64)
    depth = torch.randn(2,4,64,64)
    prp = torch.randn(2,34)

    model = FeatureExtractor(0,0,34)
    data = model(depth,rgb,prp)
    print(data.shape)