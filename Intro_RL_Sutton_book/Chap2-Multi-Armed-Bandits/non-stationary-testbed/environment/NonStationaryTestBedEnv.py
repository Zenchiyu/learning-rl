# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen


"""

import numpy as np
import copy

class MultiArmedBanditEnv:
    """
    Parent class for all Multi Armed Bandit Environment that are nonassociative.

    This class only gives 0 as reward for all actions.
    """
    def __init__(self, horizon: int=1000, random_seed=None):
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
        
        self.horizon = horizon
        self.action_space = np.arange(10)
        
        self.reset()
    
    def reset(self):
        self.q_star = np.zeros(self.action_space.shape)
        self.t = 0
        
    def step(self, action: int):
        self.t += 1
        reward = 0
        done = self.t == self.horizon
        return reward, done


class NonStationaryTestBedEnv(MultiArmedBanditEnv):
    def __init__(self, horizon: int=10000, random_seed=None):
        """
        Initialize the environment. Non associative setting, one single
        situation/state. Non stationary problem where the reward distribution
        changes over time.

        Parameters
        ----------
        horizon : int, optional
            Number of action selections or time steps. The default is 10000.
            
        random_seed : TYPE, optional
            Random seed for all the episodes. The default is None.

        Returns
        -------
        None.

        """
        # Calls overloaded methods as well
        super().__init__(horizon=horizon, random_seed=random_seed)
    
    def reset(self):
        # (re)initialize the ten reward distributions
        # create the initial true action values for the 10 actions
        self.q_star = np.zeros(self.action_space.shape)
        # reward distributions are normal distributions centered around
        # their q_star and with unit variance, q_star changes over time.
        
        # track the time step
        self.t = 0
        
    def step(self, action: int):
        """
        Take an environment step based on the action chosen by the agent.
        In our case, it's just pulling one arm, sampling a reward based
        on the reward distribution of the arm/action.
        
        Parameters
        ----------
        action : int
            Action to perform. Arm/lever to pull

        Returns
        -------
        reward : int or float
            scalar reward signal after taking the action (pulling the arm).
        done : bool
            if the episode is finished or not.
        """
        self.t += 1
        previous_q_star_a = copy.deepcopy(self.q_star[action])
        # Independent random walk for all action values
        self.q_star += np.random.normal(loc=0, scale=0.01, size=self.q_star.shape)
        
        return np.random.normal(previous_q_star_a, 1), self.t == self.horizon

if __name__ == "__main__":
    env = NonStationaryTestBedEnv(horizon=5, random_seed=42)
   
    # One episode
    done = False
    while not done:
        action = np.random.choice(env.action_space)  # currently taking random actions
        reward, done = env.step(action)
        print(reward, done)
    
    print(f"last step true action values: {env.q_star}")
    env.reset()
    print(f"initial true action values: {env.q_star}")