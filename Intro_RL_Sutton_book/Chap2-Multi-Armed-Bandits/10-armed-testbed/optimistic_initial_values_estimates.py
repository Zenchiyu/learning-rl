# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to replicate experiments for optimistic initial values
from Sutton and Barto's book on 10 armed testbed with constant step size
parameter (to keep the permanent bias).
"""
from exploration_exploitation_functions import *
    
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 1000
    n_runs = 2000
    
    # All methods use constant step size trick with alpha=0.1 (default)
    # Optimistic Greedy : Q_init = 5, epsilon = 0
    _, _, rewardsOptimisticGreedy, N_optsOptimisticGreedy = zip(*[single_run(epsilon=0, horizon=horizon, estimation_method="exponential recency-weighted avg", Q_init=5) for _ in range(n_runs)])
    # Realistic Eps-Greedy : Q_init = 0, epsilon = 0.1
    _, _, rewardsRealisticEpsGreedy, N_optsRealisticEpsGreedy = zip(*[single_run(epsilon=0.1, horizon=horizon, estimation_method="exponential recency-weighted avg") for _ in range(n_runs)])
    # Pessimistic Greedy (not in the book) : Q_init = -5, epsilon = 0
    _, _, rewardsPessimisticGreedy, N_optsPessimisticGreedy = zip(*[single_run(epsilon=0, horizon=horizon, estimation_method="exponential recency-weighted avg", Q_init=-5) for _ in range(n_runs)])
    
    # Changing tuples of tuples into arrays
    arr_rewardsOptimisticGreedy = np.array(rewardsOptimisticGreedy)
    arr_rewardsRealisticEpsGreedy = np.array(rewardsRealisticEpsGreedy)
    arr_rewardsPessimisticGreedy = np.array(rewardsPessimisticGreedy)
    
    arr_N_optsOptimisticGreedy = np.array(N_optsOptimisticGreedy)
    arr_N_optsRealisticEpsGreedy = np.array(N_optsRealisticEpsGreedy)
    arr_N_optsPessimisticGreedy = np.array(N_optsPessimisticGreedy)
    
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
    
    # Divide by 1 2 3 ... 1000 elementwise to get the percentage correctly
    sns.lineplot(x=xs, y=arr_N_optsOptimisticGreedy.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_optsRealisticEpsGreedy.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_optsPessimisticGreedy.flatten()/(xs+1))
    plt.legend([r"$Q_1(a)=5, \epsilon=0, \alpha=0.1$", r"$Q_1(a)=0, \epsilon=0.1, \alpha=0.1$", r"$Q_1(a)=-5, \epsilon=0, \alpha=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    