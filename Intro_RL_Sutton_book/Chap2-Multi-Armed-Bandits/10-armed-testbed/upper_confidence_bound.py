# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for UCB from Sutton and Barto's
book on 10 armed testbed with sample-averages methods.
"""

from exploration_exploitation_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 1000
    n_runs = 2000
    
    # UCB c=2
    _, _, rewardsUCB_c2, N_optsUCB_c2 = zip(*[single_run(epsilon=0, horizon=horizon, action_selection_method="ucb", c=2) for _ in range(n_runs)])
    # UCB c=1
    _, _, rewardsUCB_c1, N_optsUCB_c1 = zip(*[single_run(epsilon=0, horizon=horizon, action_selection_method="ucb", c=1) for _ in range(n_runs)])
    # Eps-Greedy : epsilon = 0.1
    q_stars0dot1, Qs0dot1, rewards0dot1, N_opts0dot1 = zip(*[single_run(epsilon=0.1, horizon=horizon) for _ in range(n_runs)])
    
    # Changing tuples of tuples into arrays
    arr_rewardsUCB_c2 = np.array(rewardsUCB_c2)
    arr_rewardsUCB_c1 = np.array(rewardsUCB_c1)
    arr_rewards0dot1 = np.array(rewards0dot1)
    
    arr_N_optsUCB_c2 = np.array(N_optsUCB_c2)
    arr_N_optsUCB_c1 = np.array(N_optsUCB_c1)
    arr_N_opts0dot1 = np.array(N_opts0dot1)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    sns.lineplot(x=xs, y=arr_rewardsUCB_c2.flatten())
    sns.lineplot(x=xs, y=arr_rewardsUCB_c1.flatten())
    sns.lineplot(x=xs, y=arr_rewards0dot1.flatten())
    
    plt.legend([r"UCB $c=2$", r"UCB $c=1$", r"$\epsilon=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    
    # Divide by 1 2 3 ... 1000 elementwise to get the percentage correctly
    sns.lineplot(x=xs, y=arr_N_optsUCB_c2.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_optsUCB_c1.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_opts0dot1.flatten()/(xs+1))
    plt.legend([r"UCB $c=2$", r"UCB $c=1$", r"$\epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")