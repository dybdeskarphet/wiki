# %% [markdown]
r"""
---
title: "Z-Transformation of Binomial Sampling Distribution"
date: 2026-05-25
tags: [statistics, binomial, z-transform, normal-distribution]
description: "Analyzing whether the z-transformation of sample means of a binomial random variable yields standard normal."
---
"""


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
r"""
Let's create a binomial distribution with $p=0.7$.
"""

# %%
data = np.random.choice([1, 0], 30000, p=[0.7, 0.3])
data[:20]

# %%
data_mean = np.mean(data)
data_var = np.var(data)

# %%
data_pmf = pd.Series(data).value_counts(normalize=True).sort_index()
data_pmf

# %%
fig, ax = plt.subplots()
ax.bar(data_pmf.index, data_pmf)
ax.set_xticks([0, 1])
plt.show()

# %% [markdown]
r"""
Looks about right.
"""

# %% [markdown]
r"""
## z-transformation using "normal" sampling

As you can see, CLT also works on binomial distribution.
"""

# %%
means = [np.mean(np.random.choice(data, 100)) for _ in range(5000)]
bins = np.arange(np.min(means), np.max(means), 0.005)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(means, ax=ax[0])
sns.rugplot(means, color="red", height=0.05, ax=ax[0])
ax[1].hist(means, bins=bins, rwidth=0.9)
plt.show()

# %% [markdown]
r"""
But does bootstraping work on binomial distribution?
"""

# %%
sample = np.random.choice(data, 100)
means_bootstrap = [
    np.mean(np.random.choice(sample, len(sample), replace=True)) for _ in range(5000)
]
bins_bootstrap = np.arange(np.min(means_bootstrap), np.max(means_bootstrap), 0.005)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(means_bootstrap, ax=ax[0])
sns.rugplot(means_bootstrap, color="red", height=0.05, ax=ax[0])
ax[1].hist(means_bootstrap, bins=bins_bootstrap, rwidth=0.9)
plt.show()

# %% [markdown]
r"""
Yes, it works. Because bootstrap will work where the CLT works.
"""
