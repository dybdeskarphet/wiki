# %% [markdown]
r"""
---
title: "Probability Density Functions (PDF)"
date: 2026-05-25
tags: [statistics, think-stats, probability, pdf, kde, python]
description: "Kernel density estimation (KDE), moments, skewness, and continuous distributions."
---
"""


# %% [markdown]
r"""
# Probability Density Functions
"""

# %%
from random import sample
import matplotlib
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas._libs import interval
from scipy.stats import expon, f, trimboth, norm
from scipy.integrate import simpson
import seaborn as sns

# %% [markdown]
r"""
## Normal distribution PDF

We start with the PDF of the normal distribution. Formula of normal distribution PDF is:

$$
\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Let's evaulate range of weights (NSFG birth data) using Normal PDF.
"""

# %%
birth = pd.read_csv("./data/2002FemPreg_after_01.csv")
birth

# %%
trimmed_weight = trimboth(birth["totalwgt_kg"].dropna(), 0.01)


# %%
trim_mean, trim_std = np.mean(trimmed_weight), np.std(trimmed_weight)
trim_high, trim_low = trim_mean + trim_std * 4, trim_mean - trim_std * 4

# %%
x_ticks = np.linspace(trim_low, trim_high, 201)

# %%
norm_dist = norm.pdf(x_ticks, trim_mean, trim_std)
norm_pdf = pd.Series(norm_dist, index=x_ticks)
norm_pdf

# %%
fig, ax = plt.subplots()
ax.plot(norm_pdf.index, norm_pdf)
plt.show()

# %% [markdown]
r"""
We know that area under PDF equals to 1, let's test that too.
"""

# %%
simpson(x=x_ticks, y=norm_dist)

# %% [markdown]
r"""
If we wanted to find the fraction of birth weights between 1 and 2 kgs, we could do:
"""

# %%
x_between_1_2 = np.linspace(1, 2, 201)
y_between_1_2 = norm.pdf(x_between_1_2, trim_mean, trim_std)
simpson(x=x_between_1_2, y=y_between_1_2)

# %% [markdown]
r"""
We can also find the fraction by $F_X(2) - F_X(1)$
"""

# %%
norm.cdf(2, trim_mean, trim_std) - norm.cdf(1, trim_mean, trim_std)

# %% [markdown]
r"""
## Kernel Density Estimation (KDE)

We can use KDE to estimate a PDF from the discrete data.
"""

# %%
birth_weights = birth["totalwgt_kg"].dropna()
trimmed_weight = pd.Series(trimboth(birth_weights, 0.02))

# %%
np.random.seed(3)
n = 500
sample_birth_weights = trimmed_weight.sample(n)
sample_birth_weights

# %%
weight_kde_pmf = pd.Series(1 / n, index=sample_birth_weights)
weight_kde_pmf
len(weight_kde_pmf)

# %% [markdown]
r"""
We can find the optimal $h$ using _Scott's rule of thumb_.

$$
h \approx 1.06 \cdot \hat{\sigma}n^{\frac{1}{5}}
$$
"""

# %%
x_vals = np.linspace(0, 7, 1000)
h = 1.06 * sample_birth_weights.std() * len(sample_birth_weights) ** (-1 / 5)
h

# %%
norm_dist = norm.pdf(
    x_vals, loc=sample_birth_weights.mean(), scale=sample_birth_weights.std()
)
norm_pdf = pd.Series(norm_dist, index=x_vals)
norm_pdf

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(weight_kde_pmf.index, weight_kde_pmf, width=0.01, color="red", alpha=0.6)
kde_total = np.zeros_like(x_vals)
for point in sample_birth_weights:
    kernel = norm.pdf(x_vals, loc=point, scale=h) * (1 / n)
    ax[1].plot(x_vals, kernel, color="gray", alpha=0.5, linestyle="-", linewidth=0.1)
    kde_total += kernel

ax[2].plot(norm_pdf.index, norm_pdf, color="blue", alpha=0.5, label="normal dist")
ax[2].plot(x_vals, kde_total, color="red", label="kde")

ax[0].set_xlabel("Weight")
ax[0].set_ylabel(f"1/{n}")
ax[0].set_title("First step - Data points for each weight")

ax[1].set_title("Second Step - Normal distribution PDFs for each weight")

ax[2].set_xlabel("Weight (kg)")
ax[2].set_ylabel("Probability")
ax[2].set_title("KDE plot of weights")
ax[2].legend()


plt.show()

# %% [markdown]
r"""
We have our KDE plot. In some cases, like here where we have too many data points, it is better to use KDE plot instead of a histogram.

We can calculate the area below our KDE plot to ensure that our KDE plot is also a PDF.
"""

# %%
simpson(x=x_vals, y=kde_total)

# %% [markdown]
r"""
## The Distribution Framework

At this point, we have a complete set of ways to represent distributions: PMFs, CDFs and PDFs. Now, let's learn how to transform one to another.

<img src="/notes/ts/assets/distribution_framework.png" height="300"/>
"""

# %% [markdown]
r"""
Let's practice these transformations and representations using the popular "44 babies in one 24-hour period" dataset.
"""

# %%
colspecs = [(1, 8), (9, 16), (17, 24), (25, 32)]
column_names = ["time", "sex", "weight_g", "minutes"]
boom = pd.read_fwf(
    "./data/babyboom.dat", colspecs=colspecs, names=column_names, skiprows=59
)
boom

# %%
intervals = boom["minutes"].diff().dropna()
intervals

# %%
intervals_pmf = intervals.value_counts(normalize=True).sort_index()
intervals_pmf

# %%
fig, ax = plt.subplots()
ax.bar(intervals_pmf.index, intervals_pmf)
plt.show()

# %%
intervals_cdf = intervals_pmf.cumsum()
intervals_cdf

# %%
fig, ax = plt.subplots()
ax.step(intervals_cdf.index, intervals_cdf)
plt.show()

# %%
intervals_pmf_from_cdf = intervals_cdf.diff().fillna(intervals_cdf.iloc[0])
intervals_pmf_from_cdf

# %%
fig, ax = plt.subplots()
ax.bar(intervals_pmf_from_cdf.index, intervals_pmf_from_cdf, alpha=0.4)
ax.bar(intervals_pmf.index, intervals_pmf, alpha=0.4)
plt.show()

# %% [markdown]
r"""
We can use `.allclose()` to check if there are any differences between the first and the second intervals PMF.
"""

# %%
np.allclose(intervals_pmf, intervals_pmf_from_cdf)

# %% [markdown]
r"""
We converted a PMF to CDF, then CDF to PMF, now let's create a KDE plot using the PMF.
"""

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(intervals, ax=ax[0], bw_method="scott")
ax[1].bar(intervals_pmf.index, intervals_pmf)
ax[0].set_xlim(-5, 165)
plt.show()

# %% [markdown]
r"""
Let's see how the CDF looks again.
"""

# %%
intervals_cdf = intervals_pmf.cumsum()
intervals_cdf

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(intervals_cdf.index, intervals_cdf)
plt.show()

# %% [markdown]
r"""
It looks exponential CDF.
"""

# %%
x_ticks = np.arange(0, np.max(intervals))
print(x_ticks)

# %%
params = expon.fit(intervals)
exp_cdf = pd.Series(
    expon.cdf(x=x_ticks, loc=intervals.min(), scale=intervals.std()),
    index=x_ticks,
)
exp_cdf

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(exp_cdf.index, exp_cdf, linestyle=":", color="gray")
ax.step(intervals_cdf.index, intervals_cdf)
plt.show()

# %% [markdown]
r"""
it looks like an exponential cdf, but it doesn't really fit.
"""

# %% [markdown]
r"""
## Exercises

### World cup
"""

# %%
x_ticks = np.linspace(0, 1, 1000)
first_goal = pd.Series(expon.pdf(x=x_ticks, scale=(1 / 2.5)), index=x_ticks)
first_goal
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(first_goal.index, first_goal)
ax.set_ylabel("probability")
ax.set_xlabel("in games (in this case, for 1 game)")
plt.show()

# %% [markdown]
r"""
to calculate the first goal getting scored by the halftime, we can use `simpson` again.
"""

# %%
x_halftime = np.linspace(0, 0.5, 1000)
y_halftime = expon.pdf(x=x_halftime, scale=(1 / 2.5))
first_goal_halftime_prob = simpson(x=x_halftime, y=y_halftime)

# %% [markdown]
r"""
This means that, probability of first goal happening in the first 45 minutes is 71.34. Let's show it in the distribution too.
"""

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(first_goal.index, first_goal)
ax.set_ylabel("probability")
ax.set_xlabel("in games (in this case, for 1 game)")
ax.fill_between(
    x_halftime,
    y_halftime,
    alpha=0.3,
    label=f"{round(first_goal_halftime_prob,3)*100}%",
)
ax.legend()
plt.show()

# %% [markdown]
r"""
We can also use a CDF to calculate this.
"""

# %%
expon_cdf = pd.Series(expon.cdf(x=x_ticks, scale=(1 / 2.5)), index=x_ticks)
halfway = expon_cdf.index.get_indexer([0.5], method="nearest")[0]

# %%
x_ticks = np.linspace(0, 1, 2000)
first_goal_cdf = pd.Series(expon_cdf, index=expon_cdf.index)
first_goal_cdf

# %%
fig, ax = plt.subplots()
ax.vlines(
    x=0.5, ymin=0, ymax=expon_cdf.iloc[halfway], linestyle="--", color="lightgray"
)
ax.hlines(expon_cdf.iloc[halfway], xmax=0.5, xmin=0, linestyle="--", color="lightgray")
ax.plot(first_goal_cdf.index, first_goal_cdf)
ax.set_xlim(0)
ax.set_ylim(0)
plt.show()
# %%


# %% [markdown]
r"""
## Glossary from the resource

- **continuous:** A quantity is continuous if it can have any value in a range on the number line. Most things we measure in the world – like weight, distance, and time – are continuous.
- **discrete:** A quantity is discrete if it can have a limited set of values, like integers or categories. Exact counts are discrete, as well as categorical variables.
- **probability density function (PDF):** A function that shows how density (not probability) is spread across the values of a continuous variable. The area under the PDF within an interval gives the probability that the variable falls in that interval range.
- **probability density:** The value of a PDF at a specific point; it’s not a probability itself, but it can be used to compute a probability.
- **kernel density estimation (KDE):** A method for estimating a PDF based on a sample.
- **discretize:** To approximate a continuous quantity by dividing its range into discrete levels or categories.
"""
