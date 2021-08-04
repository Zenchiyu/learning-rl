# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for greedy versus epsilon greedy
methods from Sutton and Barto's book on 10 armed testbed with sample-averages
methods.
"""
from exploration_exploitation_functions import *
from multiprocessing_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # no reproducibility with parallel executions
    horizon = 1000
    n_runs = 2000
    
    pool = init_pool()
    
    # Greedy : epsilon = 0
    _, _, rewardsGreedy, p_optsGreedy = multiple_runs(pool, n_runs, horizon=horizon, epsilon=0)
    # Eps-Greedy : epsilon = 0.01
    _, _, rewards0dot01, p_opts0dot01 = multiple_runs(pool, n_runs, horizon=horizon, epsilon=0.01)
    # Eps-Greedy : epsilon = 0.1
    _, _, rewards0dot1, p_opts0dot1 = multiple_runs(pool, n_runs, horizon=horizon, epsilon=0.1)
    
    # Changing tuples of tuples into arrays
    arr_rewardsGreedy = np.array(rewardsGreedy)
    arr_rewards0dot01 = np.array(rewards0dot01)
    arr_rewards0dot1 = np.array(rewards0dot1)
    
    arr_p_optsGreedy = np.array(p_optsGreedy)
    arr_p_opts0dot01 = np.array(p_opts0dot01)
    arr_p_opts0dot1 = np.array(p_opts0dot1)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    # https://seaborn.pydata.org/generated/seaborn.lineplot.html
    # https://www.reddit.com/r/reinforcementlearning/comments/gnvlcp/way_to_plot_goodlooking_rewards_plots/?utm_source=amp&utm_medium=&utm_content=post_body
    sns.lineplot(x=xs, y=arr_rewardsGreedy.flatten())
    sns.lineplot(x=xs, y=arr_rewards0dot01.flatten())
    sns.lineplot(x=xs, y=arr_rewards0dot1.flatten())
    
    plt.legend([r"$\epsilon=0$", r"$\epsilon=0.01$", r"$\epsilon=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    
    sns.lineplot(x=xs, y=arr_p_optsGreedy.flatten())
    sns.lineplot(x=xs, y=arr_p_opts0dot01.flatten())
    sns.lineplot(x=xs, y=arr_p_opts0dot1.flatten())
    plt.legend([r"$\epsilon=0$", r"$\epsilon=0.01$", r"$\epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")