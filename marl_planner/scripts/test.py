from pettingzoo.mpe import simple_spread_v3
import numpy as np
env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)

observation, infos = env.reset(seed=42)
print(observation)
print(env.observation_space(env.agents[0]).shape)
obs_n = []

while env.agents:
    action = {agent:env.action_space(agent).sample() for agent in env.agents}
    next_observation,rwd,termination,truncations,info = env.step(action)


print(termination)
print(truncations)