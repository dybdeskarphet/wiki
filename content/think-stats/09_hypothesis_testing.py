# %% [markdown]
r"""
---
title: "Hypothesis Testing"
date: 2026-05-25
tags: [statistics, think-stats, hypothesis-testing, p-value, python]
description: "Classical and simulation-based hypothesis testing, null hypothesis, and p-values."
---
"""


# %% [markdown]
r"""
# Hypothesis Testing

We are going to simulate 250 coin tosses, and try to get an idea of hypothesis testing.

## Flipping coins

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation as mad
from statadict import parse_stata_dict

# %% [markdown]
r"""
We create our simulation function first.
"""


# %%
def simulate_flips(n):
    return np.random.choice(["H", "T"], size=n)


# %% [markdown]
r"""
Then we simulate 250 coin flips, substract it from the expected value. And do all these steps 1000 times.
"""

# %%
simulated_states = [
    ((250 * 0.5) - np.count_nonzero(simulate_flips(250) == "H")) for i in range(1001)
]

# %% [markdown]
r"""
As you can see, MAD is not much for a fair coin flip simulation.
"""

# %%
mad(simulated_states)

# %%
heads = 140
tails = 110
mad_of_experiment = np.abs((250 * 0.5) - heads)
mad_of_experiment

# %%
fig, ax = plt.subplots()
sns.kdeplot(simulated_states, ax=ax)
ax.set_title("Deviation of a coin flip")
plt.show()

# %% [markdown]
r"""
For an experiment where 140 out of 250 coin tosses came up heads, MAD increases. What does it mean? We don't know yet, but we're going to learn it in the next section.
"""

# %% [markdown]
r"""
For this chapter, the only important part is to grasp that hypothesis test is a process consists of certain steps.

1. We start with an observation (140H, 110T), and the hypothesis that coin is biased.
2. We choose a test statistic that quanitifes the size of the observed effect. In this example, it's the absolute deviation from the expected outcome.
3. We define a null hypothesis, which is a model based on the assumption that the observed effect is due to chance, in other word, coin is fair.
4. Next, we compute the p-value, which is the probability of seeing the observed effect if the null hypothesis is true.

So we only need: **a test statistic**, **a null hypothesis**, and a **p-value** to formulate a hypothesis test.
"""

# %% [markdown]
r"""
## Testing a difference in means

Welcome our good old NSFG dataset.
"""

# %%
preg = pd.read_csv("./data/2002FemPreg.csv.xz", compression="xz")
preg

# %% [markdown]
r"""
> Will continue
"""
