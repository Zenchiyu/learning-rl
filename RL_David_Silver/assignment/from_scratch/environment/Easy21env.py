# -*- coding: utf-8 -*-
"""

@author: Stephane Liem Nguyen


"""

import numpy as np
import copy


class Easy21:
    def __init__(self, random_seed=None, verbose=True):
        """
        

        Parameters
        ----------
        random_seed : TYPE, optional
            Random seed for . The default is None.

        Returns
        -------
        None.

        """
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
            
        
        self.dict_actions = {0 : "hit", 1: "stick"}
        self.inv_dict_actions = {"hit": 0, "stick": 1}
        
        self.agent_state_space = 0
        self.action_space = np.array(list(self.inv_dict_actions.keys()))
        
        
        self.card_values = np.arange(1, 10 + 1)  # [1, 10]
        self.observation_space = {"dealer's first card": self.card_values, "player's sum": np.arange(-9, 31 + 1)}
        # the range for the player's sum includes going bust..
        
        self.dict_observation_idx = {"dealer's first card": 0, "player's sum": 1}
        
        self.initial_observation = np.vstack([np.random.choice(self.card_values), np.random.choice(self.card_values)])
        self.reset()
        
        self.verbose = verbose
    
    def reset(self):
        # The player and the dealer draw one back card each
        self.observation = copy.deepcopy(self.initial_observation)
        # What the agent does not see (he only sees the first card)
        self.dealer_sum = copy.deepcopy(self.observation[self.dict_observation_idx["dealer's first card"]])
        
    def draw_card(self):
        p_red = 1/3
        p_black = 1-p_red
        return ( 2*(np.random.uniform() < p_black) - 1 )*np.random.choice(self.card_values)
    
    def step(self, action):
        # Action already chosen by the agent
        
        info = {}
        outcomes = ["lose", "draw", "win", "bust"]
        
        if action == 0:  # hit
            
            # Player hit
            if self.verbose: print("Player hits")

            self.observation[self.dict_observation_idx["player's sum"]] += self.draw_card()
            
            player_sum = self.observation[self.dict_observation_idx["player's sum"]]
            if ((player_sum > 21) or (player_sum < 1)):
                info["outcome"] = "bust"
                if self.verbose: print("Player went bust")
                return self.observation, -1, True, info
            
            return self.observation, 0, False, info
            
        else:  # stick, the dealer hit or stick etc.
            # player sum won't change        
            player_sum = self.observation[self.dict_observation_idx["player's sum"]]
            if self.verbose: print("Player sticks")
            
            while True:
                if (self.dealer_sum >= 17): # dealer sticks
                    info["outcome"] = outcomes[int(np.sign(player_sum - self.dealer_sum) + 1)]
                    return self.observation, int(np.sign(player_sum - self.dealer_sum)), True, info
                # dealer hits
                if self.verbose: print("Dealer hits")
                self.dealer_sum += self.draw_card()
                if self.verbose: print("Dealer's sum: ", self.dealer_sum)
            
                if ((self.dealer_sum > 21) or (self.dealer_sum < 1)):
                    info["outcome"] = "win"  # bust by dealer
                    if self.verbose: print("Dealer went bust")
                    return self.observation, 1, True, info

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
    
    