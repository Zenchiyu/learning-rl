# 10-armed Testbed
We use a suite of test problems to compare numerically the effectiveness of greedy or epsilon greedy action-value methods.

We generate randomly 2000 independent different 10-armed bandit problems (their reward distributions can be different but for each bandit problem, the reward distribution does not change over time, the problem is stationary.)

For each bandit problem, we generate 10 true action values from a normal distribution of mean 0 and unit variance (standard normal distribution) and each reward distribution is a normal distribution of unit variance and centered on action values previously generated.

We also limit the number of action selections (time steps or horizon) to 1000.

## Average reward of epsilon-greedy action-value methods
The filled areas are for the [95% confidence interval](https://seaborn.pydata.org/generated/seaborn.lineplot.html)
![image](https://user-images.githubusercontent.com/49496107/126384738-b9088551-81cb-4ee0-8566-da730a40a595.png)

## Percentage of optimal action (Work In Progress)
The probability of taking the optimal action is the probability of : taking a greedy action or (taking a random action and taking the optimal action). In other words: 1 - eps + eps*eps (this formula only works epsilon is not 0)

This below is not like in Sutton's book.
![image](https://user-images.githubusercontent.com/49496107/126391445-b54a0430-7a8f-46b2-9a9d-178f70ba139c.png)

## Remark(s) about implementation
In the implementation, we estimated the true action values (mean of the reward distribution for each action) by computing the empirical mean incrementally (by sample-averages).
