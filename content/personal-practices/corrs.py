# %% [markdown]
r"""
---
title: "Pearson vs. Spearman Correlation"
date: 2026-10-17
tags: [statistics, binomial, correlation, pearson, spearman]
description: "What is the difference between Pearson correlation and Spearman correlation."
---
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

# %% [markdown]
r"""
## Pearson Correlation
First, let’s create two datasets. The definitions of our datasets are as follows:

$$t_i = \frac{1000}{99}(i - 1), \quad \text{for } i \in \{1, 2, \dots, 100\}$$

$$X_i \sim \mathcal{N}(t_i, 4)$$

$$Y_i \sim \mathcal{N}(t_i, 40000)$$
"""

# %%
data_temp = np.linspace(0,1000,100)

# %%
data_x = data_temp + np.random.normal(loc=0.0, scale=2.0, size=data_temp.size)
data_y = data_temp + np.random.normal(loc=0.0, scale=200.0, size=data_temp.size)

# %%
print(data_x[:10])
print(data_y[:10])

# %% [markdown]
r"""
It's not necessary, but let's take a look at how the data appears in `kdeplot` anyway.
"""

# %%
fig, ax = plt.subplots()
sns.kdeplot(data_x, ax=ax, color="red")
sns.kdeplot(data_y, ax=ax, color="blue")
plt.show()

# %% [markdown]
r"""
In their current state, `data_x` and `data_y` are not very different from each other. It is going to be hard to see the difference between Pearson and Spearman correlation. One of the solutions can be introducing an outlier to one of the datasets.

No matter where we add this outlier, nothing would change. Because instead of this outlier matching up with a nice negative value, and all the positive correlation suddenly being balanced out by a nice negative correlation, resulting in a value of 0, we’re directly manipulating the standard deviation.

As you may recall, when calculating the z-score, we divide by the standard deviation. And if we have an outlier, our standard deviation will be very high, so all the z-scores we obtain from `data_y` will be very small.
"""

# %%
data_y[50] = 1_000_000_000

# %% [markdown]
r"""
As we might recall, the Pearson correlation was the mean of the products of the z-normalized values of two data sets. That is:

$$r = \frac{1}{n-1} \sum_{i=1}^{n} z_{x_i} z_{y_i}$$
"""

# %%
z_x = (data_x - data_x.mean()) / data_x.std(ddof=1)
z_y = (data_y - data_y.mean()) / data_y.std(ddof=1)
pearson_r = np.sum(z_x * z_y) / (data_temp.size - 1)
pearson_r

# %% [markdown]
r"""
_Pearson r_ is very much effected from the outlier we introduced earlier. And by just looking at it, we may even think there is no correlation between these two datasets.
"""

# %% [markdown]
r"""
## Spearman Correlation

Before moving on to Spearman’s correlation, we first need to understand what the ranks of the data represent.

Let’s suppose we have a dataset consisting of the numbers [1000, 4000, 2000]. If you were to describe the order of these data points from the beginning, you’d say they’re in the 1st, 3rd, and 2nd positions, right? That’s exactly what a rank represents.

In Spearman’s correlation, we don’t use the data points themselves but rather their ranks converted into a dataset.
"""

# %%
rank_x = rankdata(data_x)
rank_y = rankdata(data_y)

# %% [markdown]
r"""
The rest is the same as with the Pearson correlation; when we calculate the Pearson correlation of the ranks of the dataset rather than the dataset itself, we obtain the Spearman correlation. To show this mathematically, as above:

$$r_s = \frac{1}{n-1} \sum_{i=1}^{n} z_{R(x_i)} z_{R(y_i)}$$
$$R(x_i) = 1 + \sum_{j \neq i} \mathbb{I}_{x_j < x_i} + \frac{1}{2} \sum_{j \neq i} \mathbb{I}_{x_j = x_i}$$
"""

# %%
print(rank_x[:5])
print(rank_y[:5])

z_rank_x = (rank_x - rank_x.mean()) / rank_x.std(ddof=1)
z_rank_y = (rank_y - rank_y.mean()) / rank_y.std(ddof=1)
spearman_corr = np.sum(z_rank_x * z_rank_y) / (data_temp.size - 1)
print(f"Pearson r: {pearson_r}\nSpearman cor.: {spearman_corr}")

# %% [markdown]
r"""
As we can see, although the Pearson correlation suggests that the datasets are largely unrelated, the more robust Spearman method indicates that our datasets are not actually that unrelated.

The reason for this is that even if the outlier value we have is 1,000,000,000 or even $10^{100}$, the rank will still fall at the end of the dataset, that is, at `100`.
"""
