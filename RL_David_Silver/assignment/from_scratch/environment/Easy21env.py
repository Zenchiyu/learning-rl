# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen


"""

import numpy as np
import copy


class Easy21:
    def __init__(self, random_seed=None, verbose=True):
        """
        Initialize the environment

        Parameters
        ----------
        random_seed : TYPE, optional
            Random seed for all the games. The default is None.
            
        verbose : TYPE, optional
            Print some informations. The default is True.

        Returns
        -------
        None.

        """
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
            
        
        self.dict_actions = {0 : "hit", 1: "stick"}
        self.inv_dict_actions = {"hit": 0, "stick": 1}
        
        self.action_space = np.array(list(self.inv_dict_actions.keys()))
        
        
        self.card_values = np.arange(1, 10 + 1)  # [1, 10]
        self.observation_space = {"dealer's first card": self.card_values, "player's sum": np.arange(-9, 31 + 1)}
        # the range for the player's sum includes going bust..
        
        self.dict_observation_idx = {"dealer's first card": 0, "player's sum": 1}
        
        # Both players draw one black card initially
        self.initial_observation = np.vstack([np.random.choice(self.card_values), np.random.choice(self.card_values)])
        self.reset()  # init or reset observation and dealer sum
        
        self.verbose = verbose
    
    def reset(self):
        """
        Init or reset observation and dealer sum

        Returns
        -------
        None.

        """
        # The player and the dealer draw one black card each
        self.observation = copy.deepcopy(self.initial_observation)
        # What the agent does not see (he only sees the first card)
        self.__dealer_sum = copy.deepcopy(self.observation[self.dict_observation_idx["dealer's first card"]])
        
    def draw_card(self):
        """
        Draw a black card or red card with probability 2/3 or 1/3 respectively.
        
        Returns
        -------
        Integer in {1, ..., 10} for black card or {-1, ..., -10} for red card.

        """
        p_red = 1/3
        p_black = 1-p_red
        return ( 2*(np.random.uniform() < p_black) - 1 )*np.random.choice(self.card_values)
    
    def goes_bust(self, card_sum : int):
        """
        Check if the "player" (dealer or player) went bust with the card_sum

        Parameters
        ----------
        card_sum : int
            card sum of either the dealer or the player.

        Returns
        -------
        bool
            If the dealer or the player went bust.

        """
        return ((card_sum > 21) or (card_sum < 1))
        
    def step(self, action: int):
        """
        Take an environment step based on the action chosen by the agent.
        
        Reward:
        -1 : losing or going bust
        0 : draw
        1 : win
        0 : otherwise
        
        
        Parameters
        ----------
        action : int
            Action to perform.

        Returns
        -------
        observation : ndarray
            dealer's first card and player's sum after taking a step.
        reward : int or float
            scalar reward signal after taking a step.
        done : bool
            if the episode is finished or not.
        info : dict
            dictionary containing informations such as the outcome of the game.

        """
        # Action already chosen by the agent
        
        info = {}
        outcomes = ["lose", "draw", "win", "bust"]
        
        if action == 0:  # If the player hits
            
            if self.verbose: print("Player hits")
            
            self.observation[self.dict_observation_idx["player's sum"]] += self.draw_card()
            player_sum = self.observation[self.dict_observation_idx["player's sum"]]
            
            # Check if player went bust after taking the action
            if self.goes_bust(player_sum):
                info["outcome"] = "bust"
                if self.verbose: print("Player went bust")
                return self.observation, -1, True, info
            
            return self.observation, 0, False, info
            
        else:  # stick, the player no longer receives cards and
            # it is the dealer's turn to hit or stick
            # player sum won't change and the case where the player goes bust
            # is impossible here
            player_sum = self.observation[self.dict_observation_idx["player's sum"]]
            if self.verbose: print("Player sticks")
            
            # Dealer's turn
            while True:
                if (self.__dealer_sum >= 17): # dealer sticks
                    # outcome is either lose, draw, win : -1, 0, 1
                    info["outcome"] = outcomes[int(np.sign(player_sum - self.__dealer_sum) + 1)]
                    return self.observation, int(np.sign(player_sum - self.__dealer_sum)), True, info
                
                # dealer hits
                if self.verbose: print("Dealer hits")
                self.__dealer_sum += self.draw_card()
                if self.verbose: print("Dealer's sum: ", self.__dealer_sum)
            
                if self.goes_bust(self.__dealer_sum):
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
    
    