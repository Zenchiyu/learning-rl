# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to do the programming Exercise 2.5 from
Sutton and Barto's book
"""
from environment.NonStationaryTestBedEnv import *
from exploration_exploitation_functions import *
from multiprocessing_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
    
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 10000  # instead of 1000 as in 10-armed testbed
    n_runs = 2000
    
    pool = init_pool()
    
    env = NonStationaryTestBedEnv(horizon=horizon)
    
    # Eps-Greedy : epsilon = 0.1 for both
    _, rewardsSampleAvg, p_optsSampleAvg = multiple_runs(pool, n_runs, env=env, horizon=horizon, estimation_method="sample-avg", epsilon=0.1)
    _, rewardsExpRecencyWeightedAvg, p_optsExpRecencyWeightedAvg = multiple_runs(pool, n_runs, env=env, horizon=horizon, estimation_method="exponential recency-weighted avg", epsilon=0.1)
    
    # Changing tuples of tuples into arrays
    arr_rewardsSampleAvg = np.array(rewardsSampleAvg)
    arr_rewardsExpRecencyWeightedAvg = np.array(rewardsExpRecencyWeightedAvg)
    
    arr_p_optsSampleAvg = np.array(p_optsSampleAvg)
    arr_p_optsExpRecencyWeightedAvg = np.array(p_optsExpRecencyWeightedAvg)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    
    sns.lineplot(x=xs, y=arr_rewardsSampleAvg.flatten())
    sns.lineplot(x=xs, y=arr_rewardsExpRecencyWeightedAvg.flatten())
    
    plt.legend([r"$\alpha_n(a)=\frac{1}{n}, \epsilon=0.1$", r"$\alpha_n(a)=0.1, \epsilon=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    sns.lineplot(x=xs, y=arr_p_optsSampleAvg.flatten())
    sns.lineplot(x=xs, y=arr_p_optsExpRecencyWeightedAvg.flatten())
    
    plt.legend([r"$\alpha_n(a)=\frac{1}{n}, \epsilon=0.1$", r"$\alpha_n(a)=0.1, \epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    