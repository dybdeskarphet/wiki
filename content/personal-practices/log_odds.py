# %% [markdown]
r"""
---
title: "Understanding Log-odds"
date: 2026-10-26
tags: [log-odds, logarithmic, likelihood, fisher-information]
description: ""
---
"""

# %% [markdown]
r"""
## Likelihood Function
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
r"""
We will use a binomial dataset and we have set the `p` to `0.7` for this example. 
"""

# %%
TRUE_P = 0.7


# %% [markdown]
r"""
We are creating two datasets. Both datasets will be configured so that there is a 0.7 probability of a result being 1 and a 0.3 probability of it being 0, but one will run this experiment 500 times while the other will run it 5,000 times.
"""

# %%
data = np.random.binomial(n=1, p=TRUE_P, size=500)
log_likelihood_reason = np.random.binomial(n=1, p=TRUE_P, size=5000)
data[:10]

# %% [markdown]
r"""
Raw likelihood function of a Bernoulli trial is:

$$
L(p) = p^x(1-p)^{1-x}
$$

If we want to calculate it for the entire dataset, it looks like this:

$$
L(p) = p^{\sum x_i}(1-p)^{n-\sum x_i}
$$
"""

# %%
def raw_likelihood(p, data):
    ones = np.sum(data == 1)
    zeros = np.sum(data == 0)
    return (p ** ones) * ((1-p) ** zeros)

# %% [markdown]
r"""
The likelihood function actually serves to provide information about what probability values are possible when we have data but TRUE_P is not known. In other words, it acts as the reverse of the probability function.

$$
\text{Probability funciton: } P(x \mid \theta)
\text{Likelihood funciton: } \mathcal{L}(\theta \mid x) = P(x \mid \theta)
$$

And as you can see below, likelihood function gives the maximum output near `0.7` which was our `TRUE_P`
"""

# %%
likelihoods = []
likelihoods_comparison = []
for p in np.linspace(start=0,stop=1,num=20):
    l = raw_likelihood(p,data)
    l_comp = raw_likelihood(p,log_likelihood_reason)
    print(f"If p={p}, Likelihood is:", l)
    likelihoods.append((p,l))
    likelihoods_comparison.append((p,l_comp))

# %% [markdown]
r"""
We can see that our dataset of 500 trials has the highest likelihood value, around 0.7. However, in our dataset of 5,000 trials, we see a meaningless line. The reason for this is floating-point underflow: when we multiply each p value by the others, we end up with numbers so small that 5,000 of them cause our program to crash.
"""

# %%
fig, axes = plt.subplots(1,2, figsize=(12,4))
x,y = zip(*likelihoods)
x_comp,y_comp = zip(*likelihoods_comparison)
axes[0].plot(x,y)
axes[1].plot(x_comp,y_comp)
plt.show()

# %% [markdown]
r"""
## Log-likelihood Function
"""

# %% [markdown]
r"""
To solve the previous problem, we could have used a library that supports operations with very small numbers. However, the raw likelihood is a function that is difficult to compute even with pen and paper.

The reason for this is that the raw likelihood is a product function:

$$L(\theta) = \prod_{i=1}^N f(x_i \mid \theta)$$

This is exactly where the logarithm comes into play. The fastest way to convert a product into a sum is to take its logarithm. The log-likelihood ensures that our function above transforms into a sum function:

$$\ln L(\theta) = \sum_{i=1}^N \ln f(x_i \mid \theta)$$
"""

# %%
def log_likelihood(p,data):
    ones = np.sum(data == 1)
    zeros = np.sum(data == 0)
    return ones * np.log(p) + zeros * np.log(1-p)


# %%
log_likelihoods = []
log_likelihoods_comparison = []
for p in np.linspace(start=0,stop=1,num=20):
    l = log_likelihood(p,data)
    l_comp = log_likelihood(p,log_likelihood_reason)
    print(f"If p={p}, Log-likelihood is:", l)
    log_likelihoods.append((p,l))
    log_likelihoods_comparison.append((p,l_comp))

# %%
fig, axes = plt.subplots(1,2, figsize=(12,4))
x,y = zip(*log_likelihoods)
x_comp,y_comp = zip(*log_likelihoods_comparison)
axes[0].plot(x,y)
axes[1].plot(x_comp,y_comp)
plt.show()

# %% [markdown]
r"""
Even though our numbers differ from those in the likelihood function, and even though our curve is different, we can see that our graph peaks around `0.7`.
"""

# %% [markdown]
r"""
## Fisher Information

So what does the sharpness of the log-likelihood curve tell us? Understanding this will actually explain what Fisher’s information means. 

Since our datasets are too far up on the page, let's create two new datasets.
"""

# %%
p_grid = np.linspace(0.05, 0.95, 1000)

# %%
data_normal = np.random.binomial(n=1, p=0.5, size=50)
data_high_size = np.random.binomial(n=1, p=0.5, size=500)

# %%
def compute_log_lik_curve(data, p_grid):
    k = np.count_nonzero(data)
    n = len(data)
    return k * np.log(p_grid) + (n - k) * np.log(1.0 - p_grid)

# %% [markdown]
r"""
First, let’s look at the plots we obtain with low n and high n.
"""

ll_curve_normal    = compute_log_lik_curve(data_normal, p_grid)
ll_curve_high_size = compute_log_lik_curve(data_high_size, p_grid)

# %%
fig, ax = plt.subplots()
ax.plot(p_grid, ll_curve_normal, label="n=50")
ax.plot(p_grid, ll_curve_high_size, label="n=500")
plt.legend()
plt.show()

# %% [markdown]
r"""
As you can see, as our sample size increases, we get an even sharper curve.

In other words, we see that the larger the dataset we have, the more directly it shapes how confident we can be in the probability value we estimate. As we move further away from 0.5 in our large sample, our log-likelihood value decreases more, but when the sample is small, it decreases slowly, meaning we are actually less confident about the `p` being `0.5`.

Now, again in a binomial dataset like this, how does the curve change if we change p?
"""

# %%
p_grid = np.linspace(0, 1, 1000)

# %%
data_half = np.random.binomial(n=1, p=0.5, size=100)
data_rigged = np.random.binomial(n=1, p=0.9, size=100)

# %%
ll_curve_half    = compute_log_lik_curve(data_half, p_grid)
ll_curve_rigged = compute_log_lik_curve(data_rigged, p_grid)

# %%
fig, ax = plt.subplots()
ax.plot(p_grid, ll_curve_half, label="p=0.5")
ax.plot(p_grid, ll_curve_rigged, label="p=0.8")
plt.legend()
plt.show()

# %% [markdown]
r"""
Mathematically, Fisher information is the negative expected value of the second derivative of the log-likelihood function with respect to a parameter, here that parameter is the true `p`.

And statistically, Fisher information measures how much evidence our data actually carries about an unknown parameter (here it is `p`).

In other words, Fisher information is a statistic that can express in a single number what the likelihood function is trying to capture.
"""

# %%
p1 = np.mean(data_normal)
p2 = np.mean(data_high_size)

info_norm = len(data_normal) / (p1 * (1 - p1))
info_size = len(data_high_size) / (p2 * (1 - p2))

print("Scenario 1 (Sample Size):")
print(f"  n=50:  Fisher Info = {info_norm:.1f}")
print(f"  n=500: Fisher Info = {info_size:.1f}")

# %%
p3 = np.mean(data_half)
p4 = np.mean(data_rigged)

info_half = len(data_half) / (p3 * (1 - p3))
info_rigged = len(data_rigged) / (p4 * (1 - p4))

print("\nScenario 2 (Location of p):")
print(f"  p=0.5: Fisher Info = {info_half:.1f}")
print(f"  p=0.9: Fisher Info = {info_rigged:.1f}")

# %% [markdown]
r"""
## The Problem with a Straight Line

We want to predict probability $p$ using a feature $x$ (e.g. hours studied).

If we use a standard linear equation:

$$p = w \cdot x + b$$

It breaks:

1. It predicts probabilities below 0 and above 1.
2. It adds the same fixed amount of probability per hour, ignoring the fact that moving $p$ near the edges requires much more evidence.
"""

# %%
x = np.linspace(0, 10, 100)
p_line = 0.2 * x - 0.5

plt.plot(x, p_line)
plt.axhline(0)
plt.axhline(1)
plt.show()

# %% [markdown]
r"""
## Log-odds

To let a linear equation add up freely, we need a scale that runs from $-\infty$ to $+\infty$.

We integrate the Bernoulli Fisher Information:

$$z = \int \frac{1}{p(1 - p)} \, dp = \ln\left(\frac{p}{1 - p}\right)$$

This is **log-odds**. 
"""

# %%
p = np.linspace(0.01, 0.99, 100)
z = np.log(p / (1 - p))

plt.plot(p, z)
plt.show()

# %% [markdown]
r"""
## Sigmoid

Now our linear model can safely predict $z$:

$$z = w \cdot x + b$$

To convert this score $z$ back into a valid probability $p$, we take the inverse:

$$p = \frac{1}{1 + e^{-z}}$$

This is the **sigmoid** function. It takes any number from $-\infty$ to $+\infty$ and squashes it into $(0, 1)$.
"""

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-6, 6, 100)
plt.plot(z_values, sigmoid(z_values))
plt.show()

# %% [markdown]
r"""
In a future article, we will discuss how all of these can be useful in logistic regression.
"""
