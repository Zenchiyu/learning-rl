# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for optimistic initial values
from Sutton and Barto's book on 10 armed testbed with constant step size
parameter (to keep the permanent bias).
"""
from exploration_exploitation_functions import *
from multiprocessing_functions import *
    
if __name__ == "__main__":
    np.random.seed(42)  # no reproducibility with parallel executions
    horizon = 1000
    n_runs = 2000
    
    pool = init_pool()
    
    # All methods use constant step size trick with alpha=0.1 (default)
    # Optimistic Greedy : Q_init = 5, epsilon = 0
    _, _, rewardsOptimisticGreedy, p_optsOptimisticGreedy = multiple_runs(pool, n_runs, horizon=horizon, estimation_method="exponential recency-weighted avg", Q_init=5, epsilon=0)
    # Realistic Eps-Greedy : Q_init = 0, epsilon = 0.1
    _, _, rewardsRealisticEpsGreedy, p_optsRealisticEpsGreedy = multiple_runs(pool, n_runs, horizon=horizon, estimation_method="exponential recency-weighted avg", epsilon=0.1)
    # Pessimistic Greedy (not in the book) : Q_init = -5, epsilon = 0
    _, _, rewardsPessimisticGreedy, p_optsPessimisticGreedy = multiple_runs(pool, n_runs, horizon=horizon, estimation_method="exponential recency-weighted avg", Q_init=-5, epsilon=0)
    
    # Changing tuples of tuples into arrays
    arr_rewardsOptimisticGreedy = np.array(rewardsOptimisticGreedy)
    arr_rewardsRealisticEpsGreedy = np.array(rewardsRealisticEpsGreedy)
    arr_rewardsPessimisticGreedy = np.array(rewardsPessimisticGreedy)
    
    arr_p_optsOptimisticGreedy = np.array(p_optsOptimisticGreedy)
    arr_p_optsRealisticEpsGreedy = np.array(p_optsRealisticEpsGreedy)
    arr_p_optsPessimisticGreedy = np.array(p_optsPessimisticGreedy)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.tile(np.arange(horizon), 2000)
    sns.lineplot(x=xs, y=arr_rewardsOptimisticGreedy.flatten())
    sns.lineplot(x=xs, y=arr_rewardsRealisticEpsGreedy.flatten())
    sns.lineplot(x=xs, y=arr_rewardsPessimisticGreedy.flatten())
    
    plt.legend([r"$Q_1(a)=5, \epsilon=0, \alpha=0.1$", r"$Q_1(a)=0, \epsilon=0.1, \alpha=0.1$", r"$Q_1(a)=-5, \epsilon=0, \alpha=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    
    sns.lineplot(x=xs, y=arr_p_optsOptimisticGreedy.flatten())
    sns.lineplot(x=xs, y=arr_p_optsRealisticEpsGreedy.flatten())
    sns.lineplot(x=xs, y=arr_p_optsPessimisticGreedy.flatten())
    plt.legend([r"$Q_1(a)=5, \epsilon=0, \alpha=0.1$", r"$Q_1(a)=0, \epsilon=0.1, \alpha=0.1$", r"$Q_1(a)=-5, \epsilon=0, \alpha=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    