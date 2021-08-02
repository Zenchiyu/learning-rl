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

def argmax_indices(Q: np.ndarray):
    """
    Return a numpy array containing all indices of the occurrences of the max.
    np.argmax only gives the first occurrence.

    Parameters
    ----------
    Q : np.ndarray
        Table containing current estimates of action values.

    Returns
    -------
    ndarray of dtype int64
        all indices of the occurrences of the max.

    """
    return np.argwhere(Q == np.max(Q)).flatten()
    
def eps_greedy(epsilon: float, Q: np.ndarray):
    """
    Epsilon-Greedy action selection with respect to Q 
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
        # greedy action with ties broken randomly
        return np.random.choice(argmax_indices(Q))
    
def single_run(epsilon: float, horizon: int, estimation_method: str="sample-avg", alpha: float=0.1):
    """
    One single run with a Nonstationary problem with time steps = horizon.

    Parameters
    ----------
    epsilon : float
        Probability of taking a random action from all possible actions.
        1-epsilon is the probability to take a greedy action.
    horizon : int
        Maximum time steps for the run.
    estimation_method : str, optional
        Method used to estimate action values. The default is "sample-avg".
        For constant step size parameter, use "exponential recency-weighted avg"
        and set alpha parameter
    alpha : float
        Constant step size parameter. The default is 0.1.
        
    Returns
    -------
    list_Q : list
        ndarrays containing estimates of action values for each time step.
    list_reward : list
        rewards for each time step.
    p_opt : ndarray of shape (horizon, )
        ndarray containing 1's and 0's with 1's at time steps where we took
        an optimal action.

    """
    env = NonStationaryTestBedEnv(horizon=horizon)
    
    # Estimated action values
    Q = np.zeros(env.action_space.shape)
    # Number of times action was taken
    N = np.zeros(env.action_space.shape)
    # Array containing 1's and 0's with 1's at time steps where we took
    # an optimal action
    p_opt = np.zeros((horizon,))
    
    list_Q = []
    list_reward = []
    
    
    done = False
    while not done:
    
        action = eps_greedy(epsilon, Q)
        reward, done = env.step(action)
        
        # Increment count
        N[action] += 1
        # If action is an optimal action (can have multiple)
        # The optimal action(s) can change over time
        if action in argmax_indices(env.q_star):
            p_opt[env.t - 1] += 1  # t-1 because t was incremented
        
        # Update estimate of action values
        if estimation_method == "sample-avg":
            Q[action] += 1/N[action]*(reward - Q[action])
        elif estimation_method == "exponential recency-weighted avg":
            Q[action] += alpha*(reward - Q[action])
        else:
            raise Exception("Estimation method does not exist.")
                
        # Record Q[action], reward: used for plotting
        list_Q += [Q[action]]
        list_reward += [reward]
        
    env.reset()
    return list_Q, list_reward, p_opt
    
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    horizon = 10000  # instead of 1000 as in 10-armed testbed
    n_runs = 2000
    
    # Eps-Greedy : epsilon = 0.1 for both
    _, rewardsSampleAvg, p_optsSampleAvg = zip(*[single_run(epsilon=0.1, horizon=horizon, estimation_method="sample-avg") for _ in range(n_runs)])
    _, rewardsExpRecencyWeightedAvg, p_optsExpRecencyWeightedAvg = zip(*[single_run(epsilon=0.1, horizon=horizon, estimation_method="exponential recency-weighted avg") for _ in range(n_runs)])
    
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
    