# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:53:41 2021

@author: Stephane Liem Nguyen

Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size eps-greedy algorithm with
eps=0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.

(p. 44)

The code is extremely slow, had to send it in Unige's cluster called baobab.
"""

from exploration_exploitation_functions import *
from multiprocessing_functions import *

if __name__ == "__main__":
    np.random.seed(42)  # no reproducibility with parallel executions
    horizon = 100000  # we use 100 000 steps instead of 200 000 because we do not
    # know why we should do more steps when the performance measure only
    # takes into account 100 000 steps
    n_runs = 2000
    
    pool = init_pool()
    env = NonStationaryTestBedEnv(horizon=horizon)
    
    # https://www.python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    params_range = [2**x for x in range(-7, 3)]
    
    params_eps_greedy = [2**x for x in range(-7, -1)]
    params_gradient_bandit = [2**x for x in range(-5, 3)]
    params_UCB = [2**x for x in range(-4, 3)]
    params_optimistic = [2**x for x in range(-2, 3)]
    
    # Fixed arguments of the algorithm and parameter ranges
    args_greedy = {"fixed_args": None, "params_algo": {"epsilon": params_eps_greedy}}
    
    # Let's use alpha as param, same as for gradient bandit..
    args_greedy_exp_recency = {"fixed_args": {"estimation_method": "exponential recency-weighted avg",\
                                              "epsilon": 0.1},\
                               "params_algo": {"alpha": params_gradient_bandit}}
    # Let's only look at the version with reward baseline using constant
    # step size param alpha
    args_gradient_bandit = {"fixed_args": {"action_selection_method": "gradient bandit", "baseline": 2},\
                            "params_algo": {"alpha": params_gradient_bandit}}
    args_UCB = {"fixed_args": {"action_selection_method": "ucb"},\
                            "params_algo": {"c": params_UCB}}
    args_optimistic = {"fixed_args": {"estimation_method": "exponential recency-weighted avg", "epsilon": 0},\
                            "params_algo": {"Q_init": params_optimistic}}
    
    # All arguments
    arguments = {r"$\epsilon$-greedy, $\alpha_n(a)=\frac{1}{n}$, param=$\epsilon$": args_greedy,\
                 r"$\epsilon$-greedy, $\epsilon=0.1$, param=$\alpha_n(a)=\alpha$": args_greedy_exp_recency,\
                 r"gradient bandit, reward baseline with constant $\alpha$, param=$\alpha$": args_gradient_bandit,\
              r"UCB, param=$c$": args_UCB,\
                  r"greedy with optimistic init. $\alpha=0.1$, param=$Q_1$": args_optimistic}
    
    # Average reward obtained over 1000 steps with a particular algorithm at a
    # particular setting of its parameter
    avgs = {r"$\epsilon$-greedy, $\alpha_n(a)=\frac{1}{n}$, param=$\epsilon$": [],\
            r"$\epsilon$-greedy, $\epsilon=0.1$, param=$\alpha_n(a)=\alpha$": [],\
            r"gradient bandit, reward baseline with constant $\alpha$, param=$\alpha$": [],\
              r"UCB, param=$c$": [],\
                  r"greedy with optimistic init. $\alpha=0.1$, param=$Q_1$": []}
    
    
    plt.figure(figsize=(20, 10))
    # https://thispointer.com/python-how-to-unpack-list-tuple-or-dictionary-to-function-arguments-using/
    for key, arguments_algo in arguments.items():
        fixed_args, params_algo = arguments_algo["fixed_args"], arguments_algo["params_algo"]
        # We suppose there's only 1 single param
        for param_name, param_values in params_algo.items():
            for param_value in param_values:
                param = {param_name: param_value}
                if fixed_args is not None:
                    _, _, rewards, p_opts = multiple_runs(pool, n_runs, env=env, horizon=horizon, **fixed_args, **param)
                else:
                    _, _, rewards, p_opts = multiple_runs(pool, n_runs, env=env, horizon=horizon, **param)
                    
                # Compute average of the 1000 steps
                # avg_rewards = np.mean(np.array(rewards), axis=0)
                # avg_p_opts = np.mean(np.array(p_opts), axis=0)
                avgs[key].append(np.mean(rewards))
            
        plt.plot(np.array(list(params_algo.values())).reshape(-1, ), avgs[key], label=key)
        
    plt.legend()
    plt.ylabel("Average reward over first 1000 time steps")
    plt.xlabel("Param")
    plt.xscale("log", base=2)