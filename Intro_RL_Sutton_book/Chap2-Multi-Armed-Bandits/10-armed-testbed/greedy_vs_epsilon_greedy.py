# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for greedy versus epsilon greedy
methods from Sutton and Barto's book on 10 armed testbed with sample-averages
methods.
"""
from exploration_exploitation_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 1000
    n_runs = 2000
    
    # Greedy : epsilon = 0
    q_starsGreedy, QsGreedy, rewardsGreedy, N_optsGreedy = zip(*[single_run(epsilon=0, horizon=horizon) for _ in range(n_runs)])
    # Eps-Greedy : epsilon = 0.01
    q_stars0dot01, Qs0dot01, rewards0dot01, N_opts0dot01 = zip(*[single_run(epsilon=0.01, horizon=horizon) for _ in range(n_runs)])
    # Eps-Greedy : epsilon = 0.1
    q_stars0dot1, Qs0dot1, rewards0dot1, N_opts0dot1 = zip(*[single_run(epsilon=0.1, horizon=horizon) for _ in range(n_runs)])
    
    # Changing tuples of tuples into arrays
    arr_rewardsGreedy = np.array(rewardsGreedy)
    arr_rewards0dot01 = np.array(rewards0dot01)
    arr_rewards0dot1 = np.array(rewards0dot1)
    
    arr_N_optsGreedy = np.array(N_optsGreedy)
    arr_N_opts0dot01 = np.array(N_opts0dot01)
    arr_N_opts0dot1 = np.array(N_opts0dot1)
    
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
    
    # Divide by 1 2 3 ... 1000 elementwise to get the percentage correctly
    sns.lineplot(x=xs, y=arr_N_optsGreedy.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_opts0dot01.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_opts0dot1.flatten()/(xs+1))
    plt.legend([r"$\epsilon=0$", r"$\epsilon=0.01$", r"$\epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")