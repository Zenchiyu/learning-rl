# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen


"""

import numpy as np
import copy

class TenArmedTestBedEnv:
    def __init__(self, horizon: int=1000, random_seed=None):
        """
        Initialize the environment. Non associative setting, one single
        situation/state.

        Parameters
        ----------
        horizon : int, optional
            Number of action selections or time steps. The default is 1000.
            
        random_seed : TYPE, optional
            Random seed for all the episodes. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
        
        self.horizon = horizon
        self.action_space = np.arange(10)
        
        # initialize the ten reward distributions
        # create the true action values for the 10 actions
        self.q_star = np.random.normal(size=(10, ))
        # reward distributions are normal distributions centered around
        # their q_star and with unit variance
        
        # reset/init : tracking the time step
        self.reset()
    
    def reset(self):
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
            scalar reward signal after taking a pulling the arm.
        done : bool
            if the episode is finished or not.
        """
        self.t += 1
        return np.random.normal(self.q_star[action], 1), self.t == self.horizon

if __name__ == "__main__":
    env = TenArmedTestBedEnv(random_seed=42)
    print(env.q_star)
    
    # One episode
    done = False
    while not done:
        action = np.random.choice(env.action_space)  # currently taking random actions
        reward, done = env.step(action)
        print(reward, done)
    
    env.reset()