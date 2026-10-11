# %% [markdown]
r"""
---
title: "Statistical Estimation"
date: 2026-05-25
tags: [statistics, think-stats, estimation, sampling, python]
description: "Estimating parameters, sample mean, variance, RMSE, and sampling distributions."
---
"""


# %% [markdown]
r"""
# Estimation
"""

# %% [markdown]
r"""
## Weighing Penguins
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from scipy.constants import pound

# %% [markdown]
r"""
Weighing Penguins
"""

# %%
sample = pd.Series(np.random.normal(0, 1, size=5000))
sample

# %%
fig, ax = plt.subplots()
ax.hist(sample, bins=np.arange(-4, 4, 0.3))
plt.show()

# %%
np.mean(sample), np.std(sample)

# %% [markdown]
r"""
We didn't have to show it in graph but we can totally say that `np.random.normal` creates normally distributed numbers.
"""


# %%
def make_normal_sample(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


# %%
ns = np.logspace(1, 5).astype(int)

# %% [markdown]
r"""
We are going to use these `n`s to see how the _estimator_ changes when the sample size increases.
"""

# %%
mu = 3.7
sigma = 0.46

# %%
means = [np.mean(make_normal_sample(mu, sigma, n)) for n in ns]
medians = [np.median(make_normal_sample(mu, sigma, n)) for n in ns]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].plot(ns, means)
axes[1].plot(ns, medians)
axes[0].set_ylabel("Sample Mean")
axes[0].set_xlabel("Sample Size")
axes[1].set_ylabel("Sample Median")
axes[1].set_xlabel("Sample Size")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
Right here, we've visualized the WLLN (Weak Law of Large Numbers) in action. We can also see that in a normal distribution, mean, median and the mode are perfectly equal in a normal distribution. But using the mean is usually preferred here because it is more statistically efficient.

**What is an estimator?** Simply put, an estimator is a rule or formula used to make an educated guess about a population parameter (like the mean, standard deviation, or median) based on sample data. When an estimator gets closer to the true population parameter as the sample size goes to infinity, we specifically call it a consistent estimator.
"""

# %% [markdown]
r"""
Before diving into deeper topics, I just want to clarify that we are currently focusing on central tendency measures. While all of these are useful under the right conditions, picking the most useful may depend on the question we’re asking. Let’s illustrate this with a real-world example.
"""

# %% [markdown]
r"""
If we plotted a histogram of global wealth, we would see a right-skewed distribution (Lognormal is our best choice here). This happens because the vast majority of people earn roughly the same average income, while a tiny fraction of the population holds an extreme amount of wealth.
"""

# %%
income = np.random.lognormal(9, 2.1, size=1_000_000)
bins = np.linspace(0, 200_000, 100)

# %%
fig, ax = plt.subplots()
ax.hist(income, bins=bins)
ax.set_xlim(0, 200_000)
ax.set_ylabel("Number of people (out of 1 mil.)")
ax.set_xlabel("Wage (dollars)")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
Let's also visualize the mean, median and the mode of this distribution.
"""

# %%
fig, ax = plt.subplots()
ax.hist(income, bins=bins)
ax.axvline(
    np.mean(income),
    linestyle="--",
    color="blue",
    alpha=0.5,
    label=f"mean ({np.mean(income):.2f})",
)
ax.axvline(
    np.median(income),
    linestyle="--",
    color="red",
    alpha=0.5,
    label=f"median ({np.median(income):.2f})",
)
ax.set_xlim(0, 200_000)
ax.set_ylabel("Number of people (out of 1 mil.)")
ax.set_xlabel("Wage (dollars)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
So comparing the mean and the median, which one would be closer to the answer we would get if we asked people about their wages? It would probably be the median value. Because the mean is very sensitive to outliers, and we have ultra-rich people in the distribution, we cannot rely on it.
"""

# %% [markdown]
r"""
So what does this have to do with estimation? Nothing. I just wanted to make it clearer that the fact we can obtain the same result using both the mean and the median above is a property of the normal distribution. Let's keep going with the estmiation.
"""

# %% [markdown]
r"""
Let's check out means and medians first.
"""

# %%
means[:5], medians[:5]

# %% [markdown]
r"""
We are sure that both mean and the median is a great estimators, but let's take a look at the RMSE (Root Mean Sqaure Error) of both estimates.
"""


# %%
def mse(estimates, actual):
    errors = np.asarray(estimates) - actual
    return np.mean(errors**2)


# %%
means_rmse = np.sqrt(mse(means, mu))
medians_rmse = np.sqrt(mse(medians, mu))
means_rmse, medians_rmse

# %% [markdown]
r"""
It looks like the RMSE of means is lower, which means the deviation of the sample means is lower compared to deviation of the sample medians.
"""

# %% [markdown]
r"""
## Robustness

The `mu` and the `sigma` we use here are from an imaginary penguin weight dataset.

If we expand on the scenario, we take the penguins, weigh them, and record their weights. However, let’s say that 2% of the penguins accidentally pressed the “unit” button, and some of them were measured in kilograms instead of pounds.
"""


# %%
def make_normal_sample_with_errors(mu, sigma, n):
    sample = np.random.normal(mu, sigma, n)
    factor = np.random.choice([1, 1 / pound], p=[0.98, 0.02], size=n)
    return factor * sample


# %%
sample = make_normal_sample_with_errors(mu, sigma, 5000)
sample_pdf = pd.Series(sample).value_counts(normalize=True)

# %%
fig, ax = plt.subplots()
ax.hist(sample, bins=40, density=True, alpha=0.8)
sns.kdeplot(sample, ax=ax, color="red")
plt.show()

# %% [markdown]
r"""
As we can see, the outliers are causing the plot to extend to the right. Let's see which estimator is unbiased with the faulty dataset.
"""

# %%
ns = np.logspace(1, 5).astype(int)
faulty_means = [np.mean(make_normal_sample_with_errors(mu, sigma, n)) for n in ns]
faulty_medians = [np.median(make_normal_sample_with_errors(mu, sigma, n)) for n in ns]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
axes[0].plot(ns, faulty_means)
axes[1].plot(ns, faulty_medians)
axes[0].axhline(mu, linestyle="--", color="red", alpha=0.6)
axes[1].axhline(mu, linestyle="--", color="red", alpha=0.6)
axes[0].set_ylabel("Sample Mean")
axes[0].set_xlabel("Sample Size")
axes[1].set_ylabel("Sample Median")
axes[1].set_xlabel("Sample Size")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
With the faulty dataset, median is the less biased estimator. That's because, as we've always been saying, the median is much more robust.
"""

# %% [markdown]
r"""
## Estimating Variance

Suppose we want to estimate the variance in the penguins' weights. But we also have to know that there are two ways of computing the variance, one is biased, and the other one is unbiased. Let's see the difference between them.
"""


# %%
def biased_var(data):
    n = len(data)
    mean = np.mean(data)
    deviations = data - mean
    return np.mean(np.sum(deviations**2) / n)


# %%
biased_vars = [biased_var(make_normal_sample(mu, sigma, n=50)) for i in range(1001)]
np.mean(biased_vars), sigma**2

# %% [markdown]
r"""
We know that our population variance is 0.2116, but `biased_vars` consistently gives an underestimation.
"""


# %%
def unbiased_var(data):
    n = len(data)
    mean = np.mean(data)
    deviations = data - mean
    return np.mean(np.sum(deviations**2) / (n - 1))


unbiased_vars = [unbiased_var(make_normal_sample(mu, sigma, n=50)) for i in range(1001)]
np.mean(unbiased_vars), sigma**2

# %% [markdown]
r"""
`unbiased_vars` gives us a much closer value to population variance. Because compared to the population, the data points are closer together in a sample, and because variance estimates deviation, we consistently overestimate the variance.

That's was the practical reason, there is also a mathematically proven reason which I'm not going to get into here.
"""

# %% [markdown]
r"""
## Sampling Distributions

So far we've been working with simulated data, now, let's see what happens with real data.
"""

# %%
peng = (
    pd.read_csv("./data/penguins_raw.csv")
    .dropna(subset="Body Mass (g)")
    .reset_index(drop=True)
)

# %%
peng.shape
peng

# %% [markdown]
r"""
For this example, we are only going to use Chinstrap penguins.
"""

# %%
chin = peng[peng["Species"] == "Chinstrap penguin (Pygoscelis antarctica)"]
chin

# %%
chin["Body Mass (kg)"] = chin["Body Mass (g)"] / 1000
chin["Body Mass (kg)"]

# %%
fig, ax = plt.subplots()
sns.kdeplot(chin["Body Mass (kg)"], ax=ax)
plt.show()

# %%
sample = chin["Body Mass (kg)"].sample(35)
np.mean(sample)

# %% [markdown]
r"""
So there are multiple methods which use resampling, we can generally use bootstrapping to estimate the population parameters, we can also use permutation tests for null hypothesis. But in this example, since we know about the population distribution, we are going to cheat and use parametric resampling, which also a form of bootstrapping but with more trust on mathematical model rather than the data.
"""


# %%
def resample(sample):
    mean, std = np.mean(sample), np.std(sample)
    return np.random.normal(mean, std, len(sample))


# %%
sample_means = [np.mean(resample(sample)) for i in range(1001)]
sample_means[:5]

# %%
np.mean(chin["Body Mass (kg)"]), np.mean(sample_means)

# %% [markdown]
r"""
This is the secnario where we try to estimate the dataset's parameters using a small part of dataset. But what if we wanted to estimate the overall parameters of the Chinstrap penguins at Antarctica, even with the ones that are not in the dataset.
"""

# %%
sample_means_overall = [np.mean(resample(chin["Body Mass (kg)"])) for i in range(10000)]
np.mean(sample_means_overall)

# %% [markdown]
r"""
Not surprisingly, we are going to see a normal distribution for the distribution of the means.
"""

# %%
fig, ax = plt.subplots()
sample_means_overall_df = pd.Series(sample_means_overall)
ax.hist(sample_means_overall_df, bins=50, alpha=0.6, edgecolor="black", density=True)
sns.kdeplot(sample_means_overall, ax=ax, color="red", alpha=0.8)
ax.set_xlim(3.51)
plt.show()

# %% [markdown]
r"""
## Standard Error

We know that standard deviation measures the spread, but to use the correct terminology, standard deviation of the sample means is called _standard error_. Let's compute that.
"""

# %%
print("Std. Err.", np.std(sample_means_overall))

# %% [markdown]
r"""
Which translates to, "if we collect many samples, we expect sample means to vary by about the above value, on average."
"""

# %% [markdown]
r"""
## Confidence Intervals

Another way of summarizing sampling distributions is to compute a confidence interval. For example, 90% confidence interval contains 90% of the values in the sampling distribution.
"""

# %%
np.percentile(sample_means_overall, [5, 95])

# %% [markdown]
r"""
## Exercies

Let's find the sampling distribution of the standard deviation for Chinstrap penguins
"""

# %%
chin


# %%
def bootstrap(sample):
    return np.random.choice(sample, len(sample), replace=True)


# %%
chin_std_distribution = [
    np.std(bootstrap(chin["Body Mass (kg)"]), ddof=1) for _ in range(10000)
]
chin_std_distribution

# %%
fig, ax = plt.subplots()
ax.hist(chin_std_distribution, bins=60)
plt.show()

# %%
np.mean(chin_std_distribution), np.std(chin["Body Mass (kg)"], ddof=1)
