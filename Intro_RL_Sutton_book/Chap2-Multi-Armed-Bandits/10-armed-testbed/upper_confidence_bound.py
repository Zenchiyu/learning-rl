# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for UCB from Sutton and Barto's
book on 10 armed testbed with sample-averages methods.
"""

from exploration_exploitation_functions import *
from multiprocessing_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # no reproducibility with parallel executions
    horizon = 1000
    n_runs = 2000
    
    pool = init_pool()
    
    # UCB c=2
    _, _, rewardsUCB_c2, p_optsUCB_c2 = multiple_runs(pool, n_runs, horizon=horizon, action_selection_method="ucb", c=2)
    # UCB c=1
    _, _, rewardsUCB_c1, p_optsUCB_c1 = multiple_runs(pool, n_runs, horizon=horizon, action_selection_method="ucb", c=1)
    # Eps-Greedy : epsilon = 0.1
    _, _, rewards0dot1, p_opts0dot1 = multiple_runs(pool, n_runs, horizon=horizon, epsilon=0.1)
    
    # Changing tuples of tuples into arrays
    arr_rewardsUCB_c2 = np.array(rewardsUCB_c2)
    arr_rewardsUCB_c1 = np.array(rewardsUCB_c1)
    arr_rewards0dot1 = np.array(rewards0dot1)
    
    arr_p_optsUCB_c2 = np.array(p_optsUCB_c2)
    arr_p_optsUCB_c1 = np.array(p_optsUCB_c1)
    arr_p_opts0dot1 = np.array(p_opts0dot1)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    sns.lineplot(x=xs, y=arr_rewardsUCB_c1.flatten())
    sns.lineplot(x=xs, y=arr_rewardsUCB_c2.flatten())
    sns.lineplot(x=xs, y=arr_rewards0dot1.flatten())
    
    plt.legend([r"UCB $c=1$", r"UCB $c=2$", r"$\epsilon=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    
    sns.lineplot(x=xs, y=arr_p_optsUCB_c1.flatten())
    sns.lineplot(x=xs, y=arr_p_optsUCB_c2.flatten())
    sns.lineplot(x=xs, y=arr_p_opts0dot1.flatten())
    plt.legend([r"UCB $c=1$", r"UCB $c=2$", r"$\epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")