## Multi-Agent Reinforcement Learning for Mobile Robots

Pytorch implementations of the multi-agent reinforcement learning algorithms, including QMIX, VDN, COMA, MADDPG, MATD3, FACMAC and MASoftQ, which are the state of the art MARL algorithms. We trained these algorithms on MPE, the Multi Particle Environments in PettingZoo. Then they are trained for path planning of swarm of mobile robots. 

### Corresponding Papers

* [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf)
* [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
* [FACMAC: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
* [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
* [Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics](https://arxiv.org/abs/1910.01465)
* [Multiagent Soft Q-Learning](https://arxiv.org/abs/1804.09817)

### Requirements

Use ```pip install -r requirements.txt``` to install the requirements.

### Quick Start

```
mkdir -p ~/marl_ws/src
cd ~/marl_ws/src
git clone https://github.com/Shaswat2001/Multi_Agent_Path_Finding.git
```

Afte that - 
```
cd ~/marl_ws
colcon build
ros2 run marl_planner main.py
```

### Results
