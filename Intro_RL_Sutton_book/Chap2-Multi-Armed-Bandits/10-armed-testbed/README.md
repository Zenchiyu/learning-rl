# 10-armed Testbed
We try to replicate a suite of test problems from Sutton and Barto's book to compare numerically the effectiveness of greedy or epsilon greedy action-value methods.

At each time step, greedy methods are choosing actions having the current highest action value estimate while epsilon greedy methods pick a random action with probability epsilon or a greedy action with one minus epsilon (Epsilon can also decrease over time but we keep it constant, see Greedy in the Limit with Infinite Exploration for more details).
Epsilon-greedy methods are trying to balance exploration and exploitation but there are other methods as well in bandit problems such as Upper Confidence Bound that we'll see later.

Action-value estimate for an action A is in our case the empirical mean of all the rewards received when we pull that arm/action A (from an underlying reward distribution associated with the action). We try to estimate the true action values, the expected reward of each arm and use the estimates to pick actions with greedy or epsilon greedy methods.

We generate randomly 2000 independent different 10-armed bandit problems (their reward distributions can be different for each bandit problem but the reward distributions for a single bandit problem do not change over time, the problem is stationary.)
All bandit problems have one single state and for each bandit problem, we generate 10 true action values from a normal distribution of mean 0 and unit variance (standard normal distribution) and each reward distribution is a normal distribution of unit variance and centered on action values previously generated.

We also limit the number of action selections (time steps or horizon) to 1000.

## Average reward of epsilon-greedy action-value methods
The filled areas are for the [95% confidence interval](https://seaborn.pydata.org/generated/seaborn.lineplot.html)
![avg reward](./images/avg_reward.PNG)

We can observe that the greedy method seems bounded and stuck. This can be due to the method taking greedily suboptimal actions most of the time if current estimates are bad for the optimal action (because did not try enough the optimal action to get a good sense of how good it is) and can potentially never update estimates of actions with low values.


## Percentage of optimal action
The probability of taking the optimal action is the probability of : taking a greedy action or (taking a random action and taking the optimal action). In other words: 1 - eps + eps*1/(number of actions) (this formula only works epsilon is not 0)

![percent optimal action](./images/percent_optimal_action.PNG)

Epsilon=0.01 increases slowly but will perform the best in the two plots.

## Remark(s) about implementation
In the implementation, we estimated the true action values (mean of the reward distribution for each action) by computing the empirical mean incrementally (by sample-averages).
To obtain the percentage of optimal action curves, we have to divide not everything by the 1000 but by each time step element-wise.
