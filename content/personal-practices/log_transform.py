# %% [markdown]
r"""
---
title: "Log-Transform"
date: 2026-10-11
tags: [statistics, logarithmic, lln, clt, normal-distribution]
description: "..."
---
"""

# %% [markdown]
r"""
## How LLN works?
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
r"""
First, let’s recall what LLN tells us. But not in a textbook sense, but based on empirical evidence.

We have a population with a uniform distribution. Let’s draw 10,000 samples of sizes 2, 10, and 50 from this population.
"""

# %%
forces_2 = np.random.uniform(low=0.0, high=1.0, size=(10000, 2))
forces_10 = np.random.uniform(low=0.0, high=1.0, size=(10000, 10))
forces_50 = np.random.uniform(low=0.0, high=1.0, size=(10000, 50))

# %% [markdown]
r"""
As we may recall, the uniform distribution looked like below.
"""

# %%
fig, ax = plt.subplots()
ax.hist(forces_10[:,1])
plt.show()

# %% [markdown]
r"""
LLN, on the other hand, tells us empirically that no matter what kind of distribution we have, as the sample size increases, the sample mean begins to approach the population mean.

Furthermore, as the sample size increases, the variance of the sampling distribution of the sample mean also decreases gradually. This aligns with the formula for the variance of this specific distribution, which we learned in statistics classes: $\frac{\sigma^2}{n}$
"""

# %%
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

axes[0].hist(forces_2.mean(axis=1), bins=50)
axes[1].hist(forces_10.mean(axis=1), bins=50)
axes[2].hist(forces_50.mean(axis=1), bins=50)

plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
## What happens when forces are multiplicative?

Imagine a simple database server pipeline with 50 sequential steps. If every step added 0 to 10ms of delay, our mean latency would be additive process.

The data we created here will have the result of those 50 steps for 10,000 pipline runs.
"""

# %%
additive_steps = np.random.uniform(low=0.0, high=10.0, size=(10000,50))

# %%
additive_latency = additive_steps.sum(axis=1)
additive_latency

# %%
fig, ax = plt.subplots()
sns.kdeplot(additive_latency, ax=ax)
plt.show()

# %% [markdown]
r"""
Here we see that our total latency is generally around 250ms. But this result has a flaw, delays are rarely an additive in real world servers, it is multiplicative. Because each step scales the payload.

Instead of defining latencies as 0-10ms for each step, we will define a base latency and multiplier between 1.0x and 1.5x.
"""

# %%
multiplicative_steps = np.random.uniform(low=1.0, high=1.5, size=(10000,50))
base_latency = 1.0
multiplicative_latency = base_latency * multiplicative_steps.prod(axis=1)

# %%
fig, ax = plt.subplots()
sns.kdeplot(multiplicative_latency, ax=ax)
fig.tight_layout()
ax.set_xlim(0)
plt.show()

# %% [markdown]
r"""
Here we can see the in the real-world scenario, latency would be much higher if multipliers were 1.0 to 1.5.

If we wanted to see the distribution better, we would make the x-scale logarithmic.
"""

# %%
fig, ax = plt.subplots()
sns.kdeplot(multiplicative_latency, ax=ax, log_scale=True)
fig.tight_layout()
plt.show()

# %% [markdown]
r"""
In the context of machine learning, working with data that is skewed in a particular direction can often lead to issues. For this reason, log transformation can make our work much easier. In this post, we won’t go into detail about what it’s used for, but it will certainly come in handy later on.
"""
