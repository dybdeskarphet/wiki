# %% [markdown]
r"""
---
title: "What Does the P-Value Mean?"
date: 2026-05-25
tags: [statistics, hypothesis-testing, p-value, simulation]
description: "Intuition, simulations, and interpretation of p-values in significance testing."
---
"""


# %%
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import seaborn as sns
from scipy.stats import norm

# %% [markdown]
r"""
A factory claims their bags of coffee beans weigh exactly 250 grams. You think they are not being honest. And take a sample.
"""

# %%
mu_null = 250
sample_mean = 245
n = 36

# %% [markdown]
r"""
You find that sample mean is 245 grams, **is it enough to say that factory is lying?**
"""

# %%
pop_std = 12

# %% [markdown]
r"""
Because we know the population standard deviation, we can easily calculate the standard deviation of the sampling distribution of the sample mean, namely, *standard error*.
"""

# %%
std_err = pop_std / np.sqrt(n)

# %% [markdown]
r"""
The key point here is that we proceed with our calculations by assuming $H_0$ is true, in a sense, we begin our reasoning with the phrase “If $H_0$ were true...”.

That is, regarding the confidence interval, we construct the normal distribution graph using the sample and its mean (since we are already discussing a specific level of confidence, within this context we placed $\bar{x}$ exactly at the center of the sampling distribution of the means), but here we construct it based on $\mu_0$.
"""

# %%
z_score = (sample_mean - mu_null) / std_err
z_score

# %% [markdown]
r"""
For this reason, we substitute $\mu_0$ for $\mu$ in the z-score formula and perform our calculations.
"""

# %% [markdown]
r"""
We found that it lies at a distance of $-2.5\sigma$. However, since this is a two-tailed test, that is, the question asks whether it differs from 250, we need to calculate the z-scores for both 245 and 255 ($250\pm5$). Although the sample mean we found indicates how much smaller the true $\mu$ value could be than 250, if we are only considering the difference, we also need to determine how much larger it could be. In other words, we are proceeding under the premise that the spread could occur in either direction.
"""

# %%
null_distribution = norm(loc=mu_null, scale=std_err)

# %%
p_value = 2 * null_distribution.cdf(sample_mean)
p_value

# %% [markdown]
r"""
Here, we found a p-value and multiplied it by 2, using the CDF of a normal distribution in the process. To understand why we did this, we first need to look at the PDF. So let’s see the visualization.
"""

# %%
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(240, 260, 1000)
ax.plot(x, null_distribution.pdf(x))
ax.axvline(sample_mean, color="red", label="sample mean", linestyle="--", alpha=0.5)
ax.axvline(
    (mu_null + (mu_null - sample_mean)),
    color="red",
    linestyle="--",
    alpha=0.5,
)

x_left_tail = np.linspace(240, sample_mean, 100)
x_right_tail = np.linspace((mu_null + (mu_null - sample_mean)), 260, 100)

ax.fill_between(x_left_tail, null_distribution.pdf(x_left_tail), color="red", alpha=0.5)
ax.fill_between(
    x_right_tail, null_distribution.pdf(x_right_tail), color="red", alpha=0.5
)

ax.legend()
plt.show()

# %% [markdown]
r"""
As we mentioned earlier, we performed a calculation based on the $\pm 5$ interval, that is, we determined how many standard deviations away the values 245 and 255 could be from the mean in a normal distribution with mean $\mu_0$, that is, a sampling distribution of the sample mean assuming the center is $\mu_0$.

Since we have a PDF, if we calculate the area shown in red, we can determine the probability that the value is 245 or less, and the probability that it is 255 or greater.

So, instead of using the formula with the CDF that we used above, if we had calculated the integral of these areas marked in red, we would still have found the P-value, that is, the probability value.
"""

# %%
p_value_alt = simpson(null_distribution.pdf(x_left_tail), x=x_left_tail)
p_value_alt += simpson(null_distribution.pdf(x_right_tail), x=x_right_tail)
p_value_alt, p_value

# %% [markdown]
r"""
It’s also worth noting that in a two-tailed test, thinking along the lines of “if we found a mean of 245, we might also find 255” can be confusing. As we recall, when performing a two-tailed calculation in a confidence interval, we added and subtracted the product of a certain standard deviation unit (z-score) multiplied by specific values, meaning we accounted for the possibility that the distribution could extend a certain number of standard deviations to the right as well as to the left.

Here, however, we can say we’re doing the opposite: since our mean is $\mu_0$, we’re accounting for the fact that 245 is actually a certain number of standard deviations to the left. In other words, in the expression $\ldots \pm z_c \cdot \frac{\sigma}{\sqrt{n}}$, we’re saying that we only have the $-$ part of the $\pm$ operation. In this example problem, we consider that the right-hand side of the expression consists only of the $-5$ part, and we include the $+5$ part ourselves, which is why we arrive at the result 255; however, viewing the situation as “... $\sigma$ away” would provide a better logical foundation.
"""

# %%
sig_interval = np.linspace(
    mu_null + norm.ppf(0.025) * (std_err), mu_null + norm.ppf(0.975) * (std_err)
)
sig_interval

# %%
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(240, 260, 1000)
ax.plot(x, null_distribution.pdf(x))
ax.axvline(sample_mean, color="orange", label="sample mean", linestyle="-", alpha=0.5)
ax.axvline(
    (mu_null + (mu_null - sample_mean)), color="orange", linestyle="-", alpha=0.5
)

x_left_tail = np.linspace(240, sample_mean, 100)
x_right_tail = np.linspace((mu_null + (mu_null - sample_mean)), 260, 100)

rejcet_area_left = np.linspace(240, null_distribution.ppf(0.025))
rejcet_area_right = np.linspace(null_distribution.ppf(0.975), 260)

ax.fill_between(
    rejcet_area_left,
    null_distribution.pdf(rejcet_area_left),
    alpha=0.5,
    color="none",
    edgecolor="red",
    hatch="//",
)
ax.fill_between(
    rejcet_area_right,
    null_distribution.pdf(rejcet_area_right),
    alpha=0.5,
    color="none",
    edgecolor="red",
    hatch="//",
    label="reject",
)

ax.fill_between(
    sig_interval,
    null_distribution.pdf(sig_interval),
    edgecolor="green",
    facecolor="none",
    alpha=0.5,
    hatch="//",
    label="fail to reject",
)

ax.axvline(
    null_distribution.ppf(0.025),
    color="green",
    alpha=0.4,
    linestyle="--",
)
ax.axvline(
    null_distribution.ppf(0.975),
    color="green",
    alpha=0.4,
    label="%5 sig. lvl.",
    linestyle="--",
)

ax.legend()
plt.show()

# %% [markdown]
r"""
Finally, when we determine that the samples we can obtain with a probability of 5% or higher are not sufficiently extreme, that is, when we set a **significance level of 5%**, we end up with a distribution like the one shown above.
"""
