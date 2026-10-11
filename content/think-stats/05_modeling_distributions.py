# %% [markdown]
r"""
---
title: "Modeling Distributions"
date: 2026-05-25
tags: [statistics, think-stats, distributions, modeling, python]
description: "Exponential, normal, lognormal, and Pareto distribution modeling and probability plots."
---
"""


# %% [markdown]
r"""
# Modeling Distributions
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
r"""
## Binomial Distribution
"""

# %% [markdown]
r"""
Let's find the distribution of skeet-shooting competition.
"""

# %%
np.random.seed(10)
flip = lambda n, p: np.random.choice([1, 0], n, p=[p, 1 - p])
pd.Series(flip(1000, 0.9)).value_counts(normalize=True)

# %% [markdown]
r"""
Here we define a function that returns us a list of ones or zeros, we can specify the probability of getting 1 with `p` parameter. Also, we can specify how many tries we want using `n` parameter.
"""

# %%
sim = lambda n, p: flip(n, p).sum()

# %% [markdown]
r"""
Let's think of ones as "the target is hit", and the zeros as "target is missed".

We defined a `sim` function where we can specify the number of tries and the probability of hittin the target, and get the successful hits out of all hits.
"""

# %%
n = 25
p = 0.9
results_sim = [sim(n, p) for i in range(1000)]
print(np.array(results_sim).mean(), n * p)

# %% [markdown]
r"""
Out of 25 tries, 22.5 of the shootings were successful on average.

But instead of simulating, we can also calculate the mean by multiplying `n` with `p`. Why? Because skeet-shooting gives us a binomial distribution. But let's compare it to a standard (textbook, no simulations) binomial distribution to see how they look side by side.
"""

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    results_sim,
    bins=range(17, 27),
    density=True,
    align="mid",
    color="skyblue",
    alpha=0.7,
    edgecolor="black",
)
ax.set_xlabel("Successful hits")
ax.set_ylabel("Probability")
ax.set_title("Simulation results")
plt.show()

# %% [markdown]
r"""
We see a distribution skewed to the left. Let's see how the textbook binomial distribution looks like for `p=0.9`.
"""

# %%
from scipy.stats import binom

# %%
print(n, p)

# %%
x_theoretical = np.arange(0, n + 1)
y_theoretical = binom.pmf(x_theoretical, n, p)
theoret_binom_freq = pd.DataFrame(y_theoretical, index=x_theoretical)
print(theoret_binom_freq)

# %%
skeet_freq = pd.Series(results_sim).value_counts(normalize=True).sort_index()
print(skeet_freq)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.4
ax.bar(
    x=skeet_freq.index - width / 2,
    height=skeet_freq,
    width=width,
    alpha=0.6,
    label="simulation",
)
ax.bar(
    x=theoret_binom_freq.index + width / 2,
    height=theoret_binom_freq[0],
    width=width,
    alpha=0.6,
    label="binomial",
)
ax.set_xlabel("Successful hits")
ax.set_ylabel("Probability")
ax.set_xlim((15, 27))
ax.legend()
plt.show()

# %% [markdown]
r"""
They look very similar. But this is no surprise since we assumed that every shooting have the same success rate.
"""

# %% [markdown]
r"""
To put our simulation in a much stronger test, we are going to use real world data.
"""

# %%
rw_skeet = pd.read_html("./data/Shooting_at_the_2020_Summer_Olympics_Mens_skeet.html")[
    6
]
rw_skeet

# %% [markdown]
r"""
Looks like `1,2,3,4,5` columns are the rounds. Let's get all the results and put it in a list.
"""

# %%
rw_skeet
rw_flat = rw_skeet[["1", "2", "3", "4", "5"]].values.flatten()

# %%
rw_success_rate = rw_flat.mean() / 25
rw_success_rate

# %% [markdown]
r"""
Real-world success rate is much higher. Let's create a binomial distribution with `p=0.953333`
"""

# %%
rw_pmf = pd.Series(rw_flat).value_counts(normalize=True).sort_index()
rw_pmf

# %%
n = 25
x_theoretical = np.arange(0, n + 1)
y_theoretical = binom.pmf(x_theoretical, n, rw_success_rate)
binom_pmf = pd.DataFrame(y_theoretical, index=x_theoretical)
print(binom_pmf)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.4
ax.bar(
    x=rw_pmf.index - width / 2,
    height=rw_pmf,
    width=width,
    alpha=0.6,
    label="actual",
)
ax.bar(
    x=binom_pmf.index + width / 2,
    height=binom_pmf[0],
    width=width,
    alpha=0.6,
    label="binomial",
)
ax.set_xlabel("Successful hits")
ax.set_ylabel("Probability")
ax.set_xlim((15, 26))
ax.legend()
plt.show()

# %% [markdown]
r"""
Binomial distribution is a good fit for the distribution of the real-world data.
"""

# %% [markdown]
r"""
## Poisson Distribution
"""

# %% [markdown]
r"""
We will simulate an 60-minute ice hockey game to understand this distribution, assuming that the teams score a total of 6 goals per game.

We can use binomial distribution as usual, let's do it.
"""

# %%
n = 3600
m = 6
p = 6 / 3600
p

# %%
goals_b = np.array([sim(n, p) for i in range(1001)])
goals_b_table = pd.Series(goals_b, np.arange(1001))
goals_b_pmf = goals_b_table.value_counts(normalize=True).sort_index()
goals_b_pmf

# %% [markdown]
r"""
We used binomial distribution perfectly, but using binomial distribution in such rare events is not ideal, I'm going to compare the success rate of getting heads and scoring in an ice-hockey game to demonstrate this.
"""

# %%
n = 1001
width = 0.5
test_coin = flip(n, 0.5)
test_goals = flip(n, p)
test_coin_sum = test_coin.sum()
test_goals_sum = test_goals.sum()

fig, ax = plt.subplots()
ax.bar(
    x=["Goals", "Coin Flips (getting heads)"], height=[test_goals_sum, test_coin_sum]
)
ax.set_ylabel("Success count")
ax.set_xlabel("Over 1000 tries")
plt.show()

# %% [markdown]
r"""
As you can see, even though we get a decent result, using binomial distribution is not ideal in this sceneario since we have another distribution for this purpose and simulating is expensive when the `n` gets larger.

Poisson distribution PMF formula looks like this:

$$
\begin{gathered}
P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!} \\
x = \text{number of success} \\
\lambda = \text{rate of success}
\end{gathered}
$$
"""

# %% [markdown]
r"""
For a binomial distribution, if $n\ge50$ and $p\le0.1$, it approximates to Poisson distribution $Po(np)$.

In the context of ice-hockey game, $\lambda = 6$, and we will have an array ranging from 0 to 20 for the $x$
"""

# %%
from scipy.stats import poisson

goals_range = np.arange(0, 20)
goals_p_tries = poisson.pmf(goals_range, m)
goals_p_pmf = pd.Series(goals_p_tries, goals_range)
goals_p_pmf

# %% [markdown]
r"""
We have our Poission PMF and simulation PMF. Let's plot them to see if Poission distribution is appropriate for this situation.
"""

# %%
fig, ax = plt.subplots()
width = 0.4

ax.bar(
    x=goals_b_pmf.index - width / 2,
    height=goals_b_pmf,
    width=width,
    alpha=0.8,
    label="simulation",
)
ax.bar(
    x=goals_p_pmf.index + width / 2,
    height=goals_p_pmf,
    width=width,
    alpha=0.8,
    label="poisson",
)
ax.set_xlabel("Number of goals")
ax.set_ylabel("Probability")
ax.legend()
plt.show()

# %% [markdown]
r"""
Yes, Poission distribution is very appropriate for this event. To put our model to a stronger test, we can use real world data.
"""

# %%
filename = "./data/nhl_2023_2024.hdf"

with pd.HDFStore(filename, "r") as store:
    keys = store.keys()

len(keys), keys[0]

# %% [markdown]
r"""
We can obtain the goals and when they were scored as follows.
"""

# %%
times = pd.read_hdf(filename, key=keys[0])
times

# %% [markdown]
r"""
This is for the first game of the season.
"""

# %% [markdown]
r"""
To get the total scores of the game, we can use `len()`
"""

# %%
len(times)

# %% [markdown]
r"""
Now let's create a list that has all the scores throughout all the games in the season.
"""

# %%
goals_real = []

for key in keys:
    goals_real.append(len(pd.read_hdf(filename, key=key)))

# %% [markdown]
r"""
Now, let's compare the real scores' PMF with the Poisson distribution PMF.
"""

# %%
goals_real_df = pd.Series(goals_real)
goals_real_pmf = goals_real_df.value_counts(normalize=True).sort_index()
goals_real_pmf

# %%
fig, ax = plt.subplots()
width = 0.4

ax.bar(
    x=goals_real_pmf.index - width / 2,
    height=goals_real_pmf,
    width=width,
    alpha=0.8,
    label="real",
)
ax.bar(
    x=goals_p_pmf.index + width / 2,
    height=goals_p_pmf,
    width=width,
    alpha=0.8,
    label="poisson",
)
ax.set_xlabel("Number of goals")
ax.set_ylabel("Probability")
ax.legend()
plt.show()

# %% [markdown]
r"""
The Poisson distribution also fits real-world data, meaning we can use the Poisson distribution to model the distribution of hockey scores.

## The Exponential Distribution
"""

# %% [markdown]
r"""
We found the average number of goals in a hockey game. But the first goal timing follows the exponential distribution. Let's see how.
"""

# %%
n = 3600
m = 6
p = m / 3600
p

# %%
np.random.seed(100)
simulate_first_goal = lambda n, p: pd.Series(flip(n, p)).argmax()
first_goal_times = [simulate_first_goal(n, p) for i in range(1001)]
first_goal_sim_cdf = (
    pd.Series(first_goal_times).value_counts(normalize=True).sort_index().cumsum()
)
first_goal_sim_cdf

# %%
mean = np.mean(first_goal_times)
mean

# %% [markdown]
r"""
You can look up formulas of the exponential distribution from its [Wikipedia page](https://en.wikipedia.org/wiki/Exponential_distribution). In this section we will use the below formulas:

$$
\begin{gathered}
\text{Mean} = \frac{1}{\lambda} \\
\text{CDF} = 1 - e^{-\lambda x}
\end{gathered}
$$

$\lambda$ represents the rate at which events occur. So we can flip the formula to find the lambda.
"""

# %%
lam = 1 / mean
lam

# %% [markdown]
r"""
Here, lambda represents how many goals are scored each second. Which is 0.0016 goal/second.
"""

# %%
first_goal_lam = 1 / mean
cdf_aggregator = lambda lam, x: 1 - np.exp(-lam * x)
cdf_size = 3600
cdf_ticks = np.linspace(0, 3600, cdf_size)
first_goal_expo_model = pd.Series(
    [cdf_aggregator(tick, lam) for tick in cdf_ticks], index=cdf_ticks
)
first_goal_expo_model

# %% [markdown]
r"""
Or we can just use scipy's `expon` function.
"""

# %%
from scipy.stats import expon

first_goal_expo_model = pd.Series(
    data=expon.cdf(cdf_ticks, scale=mean), index=cdf_ticks
)
first_goal_expo_model

# %%
fig, ax = plt.subplots()
ax.step(x=first_goal_sim_cdf.index, y=first_goal_sim_cdf, alpha=0.4, label="simulation")
ax.step(
    x=first_goal_expo_model.index, y=first_goal_expo_model, alpha=0.4, label="expon"
)
ax.legend()
plt.show()

# %% [markdown]
r"""
It matches up perfectly with the exponential distribution model.
"""

# %% [markdown]
r"""
## Normal Distribution
"""

# %% [markdown]
r"""
Most things we measure in the real world follow a normal distribution. To test this, let's consider a model of the way giant pumpkins grow.

Suppose that each day, pumpkin grows 1 pound if the weather is bad, 2 pounds if the weather is fair, and 3 pounds if the weather is good.
"""

# %%
simulate_growth = lambda n: np.random.choice([1, 2, 3], n).sum()

# %% [markdown]
r"""
We will use this function to calculate the total weight gains over `n` days.
"""

# %%
sim_weights = [simulate_growth(100) for i in range(2000)]

# %% [markdown]
r"""
Here we simulated 1001 pumpkins over 100 days. Let's also create the CDF of this `sim_wights`.
"""

# %%
sim_weights_cdf = (
    pd.Series(sim_weights).value_counts(normalize=True).sort_index().cumsum()
)
sim_cdf_size = sim_weights_cdf.count()

# %% [markdown]
r"""
For real-world plotting, plotting from negative to positive inifinity is impractical. We are only going to use the curev between $\mu - 4\sigma$ to $\mu + 4\sigma$ for this example.
"""

# %%
mean, std = np.mean(sim_weights), np.std(sim_weights)
low, high = mean - 4 * std, mean + 4 * std
print(mean, std)
print(low, high)

# %%
x_ticks = np.linspace(low, high, sim_cdf_size)
x_ticks

# %%
from scipy.stats import norm

# %%
norm_dist = norm.cdf(x_ticks, loc=mean, scale=std)
norm_dist_cdf = pd.Series(norm_dist, index=x_ticks)
norm_dist_cdf

# %% [markdown]
r"""
Let's plot them and see if the normal distribution model fits to the simulation.
"""

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.step(sim_weights_cdf.index, sim_weights_cdf, label="simulation", alpha=0.4)
ax.step(norm_dist_cdf.index, norm_dist_cdf, label="normal", alpha=0.4)
ax.legend()
plt.show()

# %% [markdown]
r"""
Because our data is discrete, it is much more appropriate to use a `.step()`, but when te sample size is small, it makes it hard to read.

In this situation, we can use the `.plot()` function the create a line plot.
"""

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sim_weights_cdf.index, sim_weights_cdf, label="simulation", alpha=0.4)
ax.plot(norm_dist_cdf.index, norm_dist_cdf, label="normal", alpha=0.4)
ax.legend()
plt.show()

# %% [markdown]
r"""
The normal distribution fits our simulation very well.

In general, when we add up enough random variables, the sum tends to follow a normal distribution, as the consequence of CLT (Central Limit Theorem).

### NSFG Data

Let's see how we see normal distribution in the real-world data.
"""

# %%
preg = pd.read_csv("./data/2002FemPreg_after_01.csv")
preg

# %% [markdown]
r"""
We will use the birth weights column.
"""

# %%
total_weights = preg["totalwgt_kg"].dropna()
print(total_weights)

# %% [markdown]
r"""
To eliminate the extreme values (outliers), we will use `trimboth`.
"""

# %%
from scipy.stats import trimboth

trimmed_weights = trimboth(total_weights, 0.01)

# %% [markdown]
r"""
Let's also create the CDF for the total weights.
"""

# %%
trimmed_weights_s = pd.Series(trimmed_weights)
trimmed_weights_cdf = (
    trimmed_weights_s.value_counts(normalize=True).sort_index().cumsum()
)

# %% [markdown]
r"""
Let's create a normal distribution with the same parameters as our weights dataset.
"""

# %%
mean, std = np.mean(trimmed_weights), np.std(trimmed_weights)
low, high = mean - std * 4, mean + std * 4

x_ticks = np.linspace(low, high, 500)
norm_dist_cdf_arr = norm.cdf(x_ticks, mean, std)
norm_dist_cdf = pd.Series(norm_dist_cdf_arr, index=x_ticks)
print(norm_dist_cdf)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(trimmed_weights_cdf.index, trimmed_weights_cdf, label="rw", alpha=0.4)
ax.plot(norm_dist_cdf.index, norm_dist_cdf, label="normal", alpha=0.4)
ax.legend()
plt.show()

# %% [markdown]
r"""
Normal model fits the data well except below 3 kilograms. It can be interpreted as lightest babies are lighter than we'd expect.
"""

# %% [markdown]
r"""
## Lognormal Distribution

We simulated pumpkin growth under the assumption that pumpkins grow 1-3 pounds per day, depending on the weather. Instead, let's suppose the growth is proportional to their current weight. So big pumpkins gain more weight per day, which is probably more realistic.
"""

# %%
sim_propo_grow = lambda n: np.random.choice([1.03, 1.05, 1.07], n).prod()
print(sim_propo_grow(20))

# %% [markdown]
r"""
Here we have a simulation of a pumpkin growing for 20 days. Let's simulate 1001 pumpkins over 100 days.
"""

# %%
sim_weights = [sim_propo_grow(100) for _ in range(1001)]
print(sim_weights)

# %% [markdown]
r"""
Let's create a CDF with this simulation results.
"""

# %%
sim_weights_cdf = (
    pd.Series(sim_weights).value_counts(normalize=True).sort_index().cumsum()
)
sim_weights_cdf

# %% [markdown]
r"""
Let's first see how this CDF looks.
"""

# %%
fig, ax = plt.subplots()
ax.plot(sim_weights_cdf.index, sim_weights_cdf)
plt.show()

# %% [markdown]
r"""
And now, let's compare it to a normal distribution with the same parameters.
"""

# %%
mean, std = np.mean(sim_weights), np.std(sim_weights)
low, high = mean - 4 * std, mean + 4 * std
print(mean, std)
print(low, high)

# %%
x_ticks = np.linspace(low, high, sim_cdf_size)
print(x_ticks)

# %%
norm_dist = norm.cdf(x_ticks, loc=mean, scale=std)
norm_dist_cdf = pd.Series(norm_dist, index=x_ticks)
norm_dist_cdf

# %%
fig, ax = plt.subplots()
ax.plot(sim_weights_cdf.index, sim_weights_cdf)
ax.plot(norm_dist_cdf.index, norm_dist_cdf)
plt.show()

# %% [markdown]
r"""
Even though we gave the function the same parameters, it is not matching. Because it is not a normal distribution, it is a log-normal distribution. Which means, logarithm of this distribution is normal distribution. Let's take a closer look.

**Note:** In our example, it doesn't really matter if we use `log10`, `ln` or `log2`. But we should consider carefully which one to use when we start working with regression models.
"""

# %%
sim_weights_log_cdf = (
    pd.Series(np.log10(sim_weights)).value_counts(normalize=True).sort_index().cumsum()
)
print(sim_weights_log_cdf)

# %%
mean_log, std_log = np.mean(np.log10(sim_weights)), np.std(np.log10(sim_weights))
low_log, high_log = mean_log - 4 * std_log, mean_log + 4 * std_log
sim_cdf_size = sim_weights_log_cdf.count()
print(mean_log, std_log)
print(low_log, high_log)

# %%
x_ticks = np.linspace(low_log, high_log, sim_cdf_size)
print(x_ticks)

# %%
norm_dist_log = norm.cdf(x_ticks, loc=mean_log, scale=std_log)
norm_dist_log_cdf = pd.Series(norm_dist_log, index=x_ticks)
norm_dist_log_cdf

# %% [markdown]
r"""
Let's see the first result where we compared the distribution to a normal distribution. And the second result where we compared the lograithm of the distribution with a normal distribution.
"""

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(sim_weights_cdf.index, sim_weights_cdf)
ax[0].plot(norm_dist_cdf.index, norm_dist_cdf)
ax[1].plot(sim_weights_log_cdf.index, sim_weights_log_cdf)
ax[1].plot(norm_dist_log_cdf.index, norm_dist_log_cdf)
ax[0].set_title("Direct comparison with normal distribution")
ax[1].set_title("Comparison with normal distribution after log10")
ax[0].set_ylabel("Cum. Prob.")
ax[1].set_ylabel("Cum. Prob.")
ax[0].set_xlabel("lbs")
ax[1].set_xlabel("log10(lbs)")
plt.show()

# %% [markdown]
r"""
If people are like pumpkins, we expect adult weights distribution to be the same. Let's test this out.
"""

# %% [markdown]
r"""
First we import our database which is from BRFSS.
"""

# %%
brfss = pd.read_csv("./data/brfss.csv", compression="gzip")
brfss

# %%
adult_weights = brfss["wtkg2"].dropna()
adult_weights
log_adult_weights = pd.Series(np.log10(adult_weights))

# %%
mean, std = adult_weights.mean(), adult_weights.std()
low, high = mean - 4 * std, mean + 4 * std
print("Mean:", mean)
print("std:", std)
print("-4 std:", low)
print("+4 std:", high)
x_ticks = np.linspace(low, high, 200)
print("X ticks", x_ticks)

# %%
norm_dist = norm.cdf(x_ticks, loc=mean, scale=std)
norm_dist_cdf = pd.Series(norm_dist, index=x_ticks)
norm_dist_cdf

# %%
adult_weights_cdf = adult_weights.value_counts(normalize=True).sort_index().cumsum()
adult_weights_cdf

# %%
fig, ax = plt.subplots()
ax.plot(adult_weights_cdf.index, adult_weights_cdf)
ax.plot(norm_dist_cdf.index, norm_dist_cdf)
plt.show()

# %% [markdown]
"""
Without taking the logarithm of the weights, normal distribution doesn't really align with it.
"""

# %% [markdown]
"""
Let's limit the x with the `high` to see it better.
"""

# %%
fig, ax = plt.subplots()
ax.plot(adult_weights_cdf.index, adult_weights_cdf)
ax.plot(norm_dist_cdf.index, norm_dist_cdf)
ax.set_xlim(right=high)
plt.show()

# %% [markdown]
"""
Another thing to notice is that the data we have is right skewed. In this distribution, taking the logarithm results with bigger values getting normalized.
"""

# %%
log_adult_weights_cdf = (
    log_adult_weights.value_counts(normalize=True).sort_index().cumsum()
)
log_adult_weights_cdf
fig, ax = plt.subplots()
ax.plot(log_adult_weights_cdf.index, log_adult_weights_cdf)
plt.show()

# %%
mean, std = log_adult_weights.mean(), log_adult_weights.std()
low, high = mean - 4 * std, mean + 4 * std
print("Mean:", mean)
print("std:", std)
print("-4 std:", low)
print("+4 std:", high)
x_ticks = np.linspace(low, high, 200)
print("X ticks", x_ticks)

# %%
norm_dist = norm.cdf(x_ticks, loc=mean, scale=std)
norm_dist_cdf = pd.Series(norm_dist, index=x_ticks)
norm_dist_cdf

# %%
fig, ax = plt.subplots()
ax.plot(log_adult_weights_cdf.index, log_adult_weights_cdf, label="log10(actual)")
ax.plot(norm_dist_cdf.index, norm_dist_cdf, label="normal_cdf")
ax.set_ylabel("Cum. Prob.")
ax.set_xlabel("log10(weight)")
ax.legend()

# %% [markdown]
"""
The normal model fits the logratihms of the weights well. Which suggests that proportional growth is a better model for weight gain.
"""

# %% [markdown]
"""
## Exercises
"""

# %%
from statadict import parse_stata_dict

fem_dict = parse_stata_dict("./data/2002FemResp.dct")
fem = pd.read_fwf(
    "./data/2002FemResp.dat", names=fem_dict.names, colspecs=fem_dict.colspecs
)

# %%
fem

# %%
fem["cmmarrhx"] = fem.cmmarrhx.replace([9997, 9998, 9999], np.nan)
fem["agemarry"] = (fem.cmmarrhx - fem.cmbirth) / 12.0
fem["age"] = (fem.cmintvw - fem.cmbirth) / 12.0

# %%
older = fem[fem["age"] >= 25]
older_numfmhh_pmf = older["numfmhh"].value_counts(normalize=True).sort_index()
mean_o_n_pmf = older["numfmhh"].mean()
older_numfmhh_pmf

# %%
fig, ax = plt.subplots()
ax.plot(older_numfmhh_pmf.index, older_numfmhh_pmf, color="blue")
plt.show()

# %%
x_ticks = np.arange(0, 8)
x_ticks
poisson_dist = poisson.pmf(k=x_ticks, mu=mean_o_n_pmf, loc=0)
poisson_pmf = pd.Series(poisson_dist, index=x_ticks)
poisson_pmf

# %%
fig, ax = plt.subplots()
ax.plot(older_numfmhh_pmf.index, older_numfmhh_pmf, color="blue")
ax.plot(poisson_pmf.index, poisson_pmf, color="red")
plt.show()

# %% [markdown]
"""
It seems that for the older female population (>=25), distribution of number of people in the household aligns with the Poisson distribution.

---
"""

# %%
adult_heights = brfss["htm3"].dropna()
adult_heights_log = pd.Series(np.log10(adult_heights))
m, s = np.mean(adult_heights), np.std(adult_heights)
m, s

# %%
mean, std = adult_heights.mean(), adult_heights.std()
low, high = mean - 4 * std, mean + 4 * std
print("Mean:", mean)
print("std:", std)
print("-4 std:", low)
print("+4 std:", high)
x_ticks = np.linspace(low, high, 200)
print("X ticks", x_ticks)

# %%
adult_heights_cdf = adult_heights.value_counts(normalize=True).sort_index().cumsum()
adult_heights_cdf

# %%
norm_dist = norm.cdf(x_ticks, loc=mean, scale=std)
norm_dist_cdf = pd.Series(norm_dist, index=x_ticks)
norm_dist_cdf

# %%
fig, ax = plt.subplots()
ax.plot(norm_dist_cdf.index, norm_dist_cdf)
ax.plot(adult_heights_cdf.index, adult_heights_cdf)
plt.show()

# %% [markdown]
r"""
It looks like normal distribution fits, but let's try log-normal distribution anyway.
"""

# %%
mean_log, std_log = adult_heights_log.mean(), adult_heights_log.std()
low_log, high_log = mean_log - 4 * std_log, mean_log + 4 * std_log
print("Mean:", mean_log)
print("std:", std_log)
print("-4 std:", low_log)
print("+4 std:", high_log)
x_ticks_log = np.linspace(low_log, high_log, 200)
print("X ticks", x_ticks_log)

# %%
adult_heights_log_cdf = (
    adult_heights_log.value_counts(normalize=True).sort_index().cumsum()
)
adult_heights_log_cdf

# %%
norm_dist_log = norm.cdf(x_ticks_log, loc=mean_log, scale=std_log)
norm_dist_log_cdf = pd.Series(norm_dist_log, index=x_ticks_log)
norm_dist_log_cdf

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(norm_dist_log_cdf.index, norm_dist_log_cdf)
ax[0].plot(adult_heights_log_cdf.index, adult_heights_log_cdf)
ax[1].plot(norm_dist_cdf.index, norm_dist_cdf)
ax[1].plot(adult_heights_cdf.index, adult_heights_cdf)
ax[0].set_title("log10")
ax[1].set_title("normal")
ax[0].set_xlim(np.log10(120))
ax[1].set_xlim(120)
plt.show()

# %% [markdown]
r"""
I couldn't find any noticable difference.
"""

# %% [markdown]
"""
## Glossary from the resource

- **binomial distribution:** A theoretical distribution often used to model the number of successes or hits in a sequence of hits and misses.
- **Poisson distribution:** A theoretical distribution often used to model the number of events that occur in an interval of time.
- **exponential distribution:** A theoretical distribution often used to model the time between events.
- **normal distribution:** A theoretical distribution often used to model data that follow a symmetric, bell-like curve.
- **lognormal distribution:** A theoretical distribution often used to model data that follow a bell-like curve that is skewed to the right.
"""
