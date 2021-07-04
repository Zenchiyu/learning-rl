# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:39:44 2021

@author: Stephane Liem Nguyen



Model-free On-Policy Monte-Carlo control

- Model-free : action-value functions instead of state-value functions
to avoid needing dynamics of system
- GLIE (greedy in the limit and infinite exploration) (exploration rate decay of eps greedy)
- Deterministic policy
- No importance sampling because on-policy
- Only on episodic tasks, use complete returns


"""


from environment.Easy21env import *


if __name__ == "__main__":
    
    # Example of one possible episode where dealer wins without player going bust
    # random_seed = 1, 3 or 12
    
    env = Easy21(1)
    print(env.observation)
    done = False
    while not done:
        action = np.random.choice([0, 1])
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
    
    