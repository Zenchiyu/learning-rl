# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:26:33 2021

@author: Stephane Liem Nguyen

In this file we try to replicate experiments for gradient bandit algorithm
with and without reward baseline on a variant of 10-armed testbed :
action values are sampled from a normal distribution of unit-variance and
around +4 instead of 0.

"""

from exploration_exploitation_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 1000
    n_runs = 2000
    loc = 4
    
    # With baseline
    # alpha=0.1
    _, _, rewardsBaselineAlpha0dot1, p_optsBaselineAlpha0dot1 = zip(*[single_run(horizon=horizon, loc=loc, alpha=0.1, action_selection_method="gradient bandit") for _ in range(n_runs)])
    # alpha=0.4
    _, _, rewardsBaselineAlpha0dot4, p_optsBaselineAlpha0dot4 = zip(*[single_run(horizon=horizon, loc=loc, alpha=0.4, action_selection_method="gradient bandit") for _ in range(n_runs)])
    
    # Without baseline
    # alpha=0.1
    _, _, rewardsNoBaselineAlpha0dot1, p_optsNoBaselineAlpha0dot1 = zip(*[single_run(horizon=horizon, loc=loc, alpha=0.1, action_selection_method="gradient bandit", baseline=False) for _ in range(n_runs)])
    # alpha=0.4
    _, _, rewardsNoBaselineAlpha0dot4, p_optsNoBaselineAlpha0dot4 = zip(*[single_run(horizon=horizon, loc=loc, alpha=0.4, action_selection_method="gradient bandit", baseline=False) for _ in range(n_runs)])
    
    # Changing tuples of tuples into arrays
    arr_rewardsBaselineAlpha0dot1 = np.array(rewardsBaselineAlpha0dot1)
    arr_rewardsBaselineAlpha0dot4 = np.array(rewardsBaselineAlpha0dot4)
    arr_rewardsNoBaselineAlpha0dot1 = np.array(rewardsNoBaselineAlpha0dot1)
    arr_rewardsNoBaselineAlpha0dot4 = np.array(rewardsNoBaselineAlpha0dot4)
    
    arr_p_optsBaselineAlpha0dot1 = np.array(p_optsBaselineAlpha0dot1)
    arr_p_optsBaselineAlpha0dot4 = np.array(p_optsBaselineAlpha0dot4)
    arr_p_optsNoBaselineAlpha0dot1 = np.array(p_optsNoBaselineAlpha0dot1)
    arr_p_optsNoBaselineAlpha0dot4 = np.array(p_optsNoBaselineAlpha0dot4)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    
    sns.lineplot(x=xs, y=arr_rewardsBaselineAlpha0dot1.flatten())
    sns.lineplot(x=xs, y=arr_rewardsBaselineAlpha0dot4.flatten())
    sns.lineplot(x=xs, y=arr_rewardsNoBaselineAlpha0dot1.flatten())
    sns.lineplot(x=xs, y=arr_rewardsNoBaselineAlpha0dot4.flatten())
    
    plt.legend([r"With baseline, $\alpha=0.1$", r"With baseline, $\alpha=0.4$", r"No baseline, $\alpha=0.1$", r"No baseline, $\alpha=0.4$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    
    sns.lineplot(x=xs, y=arr_p_optsBaselineAlpha0dot1.flatten())
    sns.lineplot(x=xs, y=arr_p_optsBaselineAlpha0dot4.flatten())
    sns.lineplot(x=xs, y=arr_p_optsNoBaselineAlpha0dot1.flatten())
    sns.lineplot(x=xs, y=arr_p_optsNoBaselineAlpha0dot4.flatten())
    plt.legend([r"With baseline, $\alpha=0.1$", r"With baseline, $\alpha=0.4$", r"No baseline, $\alpha=0.1$", r"No baseline, $\alpha=0.4$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")