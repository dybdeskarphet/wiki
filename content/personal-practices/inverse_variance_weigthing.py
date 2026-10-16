# %% [markdown]
r"""
---
title: "When is Inverse-Variance Weighting Useful?"
date: 2026-05-25
tags: [statistics, inverse-variance, estimation, meta-analysis]
description: "Derivation and practical applications of inverse-variance weighting in combining estimators."
---
"""


# %% [markdown]
r"""
## A situation where inverse-variance weighting is useless
"""

# %% [markdown]
r"""
Let's import our libraries first.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
r"""
Imagine a temperature sensor that only works occasionally and sends you the data when it's active. We can simulate this situation by having a fake data and only working with it's samples.
"""

# %%
data = np.random.normal(50, 100, 1000)
data[:5]

# %% [markdown]
r"""
Let's also see how the actual data looks like in a KDE plot.
"""

# %%
fig, ax = plt.subplots()
sns.kdeplot(data, ax=ax)
plt.show()

# %%
samples = []
for _ in range(20):
    samples.append(np.random.choice(data, 20, replace=False))

samples[:5]

# %% [markdown]
r"""
And here we have 20 samples from the population data.
"""

# %%
for i, s in enumerate(samples, 1):
    print(f"Var of {i}:", np.var(s), f"| Mean of {i}:", np.mean(s))
    print()

# %% [markdown]
r"""
We can show the mean and the variance of these samples in a beautiful `pandas` data frame. It is also going to make it easier for us to work with parameters of the samples.
"""

# %%
data_sum = pd.DataFrame(
    {
        "mean": [np.mean(s) for s in samples],
        "var": [np.var(s) for s in samples],
        "weights": [1 / np.var(s) for s in samples],
    }
)
data_sum

# %% [markdown]
r"""
Here we can also see a column called `weights`, it is calculated by taking the reciprocal of variance ($\frac{1}{\sigma^2}$). We are going to use this weights to calculate the weighted mean, which we expect to yield better results.
"""

# %%
weighted_mean = np.sum(data_sum["weights"] * data_sum["mean"]) / np.sum(
    data_sum["weights"]
)

# %%
actual_mean = np.mean(data)
print("Actual mean:", actual_mean)
print("Weighted mean:", weighted_mean)
print("Grand mean:", np.mean(data_sum["mean"]))

# %% [markdown]
r"""
As you can see in the results, _grand mean_ is giving us a better result compared to _weighted mean_. This is because we are using the same distribution, or same sensor, for sampling. When we draw samples from the same distribution, since the “uncertainity” of the samples remains the same, there is no point in using the weighted mean.
"""

# %% [markdown]
r"""
## A situation where inverse-variance weighting is useful
"""

# %% [markdown]
r"""
You can think of this actual mean as the actual temperature of a room.
"""

# %%
actual_mean = 30

# %% [markdown]
r"""
Below, we simulated different sensors with different variances (or in this context, different quailities). But because the temperature of the room cannot change from sensor to sensor, it remains the same (`actual_mean`).
"""

# %%
samples_from_different_dists = []

samples_from_different_dists.append(np.random.normal(actual_mean, 10, 10))
samples_from_different_dists.append(np.random.normal(actual_mean, 40, 10))

# Adding a lot of bad sensors to prove a point :D
samples_from_different_dists.append(np.random.normal(actual_mean, 150, 10))
samples_from_different_dists.append(np.random.normal(actual_mean, 150, 10))

# %% [markdown]
r"""
We created the same `pandas` data frame the see the results clearly. But here, difference between weights are more noticeable. Because we rely more on the low-variance sensor, while also taking the high-variance sensor's data into account.
"""

# %%
different_data_sum = pd.DataFrame(
    {
        "mean": [np.mean(s) for s in samples_from_different_dists],
        "var": [np.var(s) for s in samples_from_different_dists],
        "weights": [1 / np.var(s) for s in samples_from_different_dists],
    }
)
different_data_sum

# %%
weighted_mean_diff = np.sum(
    different_data_sum["mean"] * different_data_sum["weights"]
) / np.sum(different_data_sum["weights"])
weighted_mean_diff

# %% [markdown]
r"""
Here, the weighted mean calculated with IVW method is much closer to the actual mean than the grand mean because we are using different distributions (sensors) with different variances.
"""

# %%
print("Actual mean:", actual_mean)
print("Weighted mean:", weighted_mean_diff)
print("Grand mean:", np.mean(different_data_sum["mean"]))

# %% [markdown]
r"""
You may also ask why should we bother with high-variance data if we already have a decent low-variance data. The answer to this question is: **Data is data**, no matter what. In most scenarios, using high-variance data and giving it less weight is much more appropriate methodology than not using it at all.
"""
