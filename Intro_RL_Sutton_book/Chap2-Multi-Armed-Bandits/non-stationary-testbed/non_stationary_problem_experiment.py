# -*- coding: utf-8 -*-
"""
@author: Stephane Liem Nguyen

In this file we try to do the programming Exercise 2.5 from
Sutton and Barto's book
"""
from environment.NonStationaryTestBedEnv import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def eps_greedy(epsilon: float, Q: np.ndarray):
    """
    Greedy action selection with respect to Q 
    (one single state, situation. Non associative setting)

    Parameters
    ----------
    epsilon : float
        Probability of taking a random action from all possible actions.
        1-epsilon is the probability to take a greedy action.
    Q : np.ndarray
        Table containing current estimates of action values.

    Returns
    -------
    int
        Action selected by epsilon-greedy.

    """
    if np.random.uniform() < epsilon:
        # random action
        return np.random.choice(np.arange(Q.size))
    else:
        # greedy action
        return np.argmax(Q)
    
def single_run(epsilon: float, horizon: int, estimation_method="sample-avg", alpha=0.1):
    """
    One single run with a Nonstationary problem with time steps = horizon.

    Parameters
    ----------
    epsilon : float
        Probability of taking a random action from all possible actions.
        1-epsilon is the probability to take a greedy action.
    horizon : int
        Maximum time steps for the run.

    Returns
    -------
    np.ndarray
        Actual action-values.
    list_Q : list
        ndarrays containing estimates of action values for each time step.
    list_reward : list
        rewards for each time step.
    list_N_opt : list
        number of times optimal action was taken for each time step.

    """
    env = NonStationaryTestBedEnv(horizon=horizon)
    
    # Estimated action values
    Q = np.zeros(env.action_space.shape)
    # Number of times action was taken
    N = np.zeros(env.action_space.shape)
    # Number of times optimal action was taken
    N_opt = 0
    # The optimal action
    opt_action = np.argmax(env.q_star)
    
    list_Q = []
    list_reward = []
    list_N_opt = []
    
    
    done = False
    while not done:
        action = eps_greedy(epsilon, Q)
        reward, done = env.step(action)
        
        # Increment count
        N[action] += 1
        if action == opt_action:
            N_opt += 1
        
        # Update estimate of action values
        if estimation_method == "sample-avg":
            Q[action] += 1/N[action]*(reward - Q[action])
        elif estimation_method == "exponential recency-weighted avg":
            Q[action] += alpha*(reward - Q[action])
        else:
            raise Exception("Estimation method does not exist.")
            
        # Record Q[action], reward, N_opt : used for plotting
        list_Q += [Q[action]]
        list_reward += [reward]
        list_N_opt += [N_opt]
        
    env.reset()
    return env.q_star, list_Q, list_reward, list_N_opt
    
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 10000  # instead of 1000 as in 10-armed testbed
    n_runs = 2000
    
    # Eps-Greedy : epsilon = 0.1 for both
    _, _, rewardsSampleAvg, N_optsSampleAvg = zip(*[single_run(epsilon=0.1, horizon=horizon, estimation_method="sample-avg") for _ in range(n_runs)])
    _, _, rewardsExpRecencyWeightedAvg, N_optsExpRecencyWeightedAvg = zip(*[single_run(epsilon=0.1, horizon=horizon, estimation_method="exponential recency-weighted avg") for _ in range(n_runs)])
    
    # Changing tuples of tuples into arrays
    arr_rewardsSampleAvg = np.array(rewardsSampleAvg)
    arr_rewardsExpRecencyWeightedAvg = np.array(rewardsExpRecencyWeightedAvg)
    
    arr_N_optsSampleAvg = np.array(N_optsSampleAvg)
    arr_N_optsExpRecencyWeightedAvg = np.array(N_optsExpRecencyWeightedAvg)
    
    # Plots (takes a long time with lineplot)
    # Average Reward
    plt.figure(figsize=(20, 10))
    # Get array containing 0 1 2 3 ... 999 repeated 2000 times
    xs = np.broadcast_to(np.arange(horizon), arr_rewardsSampleAvg.shape).flatten()
    
    sns.lineplot(x=xs, y=arr_rewardsSampleAvg.flatten())
    sns.lineplot(x=xs, y=arr_rewardsExpRecencyWeightedAvg.flatten())
    
    plt.legend([r"$\alpha_n(a)=\frac{1}{n}, \epsilon=0.1$", r"$\alpha_n(a)=0.1, \epsilon=0.1$"])
    plt.ylabel("Average reward")
    plt.xlabel("Steps")
    
    
    # Optimal action
    plt.figure(figsize=(20, 10))
    # Divide by 1 2 3 ... 1000 elementwise to get the percentage correctly
    sns.lineplot(x=xs, y=arr_N_optsSampleAvg.flatten()/(xs+1))
    sns.lineplot(x=xs, y=arr_N_optsExpRecencyWeightedAvg.flatten()/(xs+1))
    
    plt.legend([r"$\alpha_n(a)=\frac{1}{n}, \epsilon=0.1$", r"$\alpha_n(a)=0.1, \epsilon=0.1$"])
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    