# %% [markdown]
r"""
---
title: "Distributions and Summary Statistics"
date: 2026-05-25
tags: [statistics, think-stats, distributions, python]
description: "Representing distributions, histograms, PMFs, and outlier detection."
---
"""


# %% [markdown]
# # Distributions

# %% [markdown]
# ## Distributions

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df = pd.read_csv("./data/2002FemPreg_after_01.csv")
df

# %% [markdown]
# This data frame has the last modifications we did at [Exploratory Data Analysis](/notebook.html?src=/notes/think_stats/01_exploratory_data_analysis.ipynb).

# %% [markdown]
# Let's create an histogram for the weight distribution.

# %%
wgt = df[df["outcome"] == 1]["totalwgt_kg"]
wgt

# %%
fig, ax = plt.subplots()

ax.hist(wgt, bins=15, color="skyblue", edgecolor="black")
plt.show()

# %% [markdown]
# This is a very recognizable bell curve. This is expected. Let's look at another important feature of the featurfeature.

# %%
wgt.mode()

# %% [markdown]
# Total weight is expected to be a normal distribution because of CLT. But nature doesn't know about the difference between ounces and pounds, especially in this context, weight split between ounces and pounds. Therefore, ounces shouldn't follow CLT as it is totally random.

# %%
wgt_oz = df["birthwgt_oz"]
wgt_oz

# %%
fig, ax = plt.subplots()

ax.hist(wgt_oz, bins=np.arange(0,17.5), edgecolor="black", color="skyblue")
plt.show()

# %% [markdown]
# As we can see, it is in fact a uniform distribution. But we can also see that *0 ounces* is more common. This is probably due to respondents rounding up the weight to an integer.

# %% [markdown]
# Let's look at the histogram of mothers' ages at the end of the birth.

# %%
mother_age = df["agepreg"]
mother_age.value_counts()

# %%
fig, ax = plt.subplots()

ax.hist(mother_age, bins=np.arange(10,50,0.6), edgecolor="black", color="skyblue")
plt.show()

# %% [markdown]
# It is roughly a normal distribution. But it is skewed to the right.

# %% [markdown]
# Let's look at the distribution of the length of the pregnancy.

# %%
preg_dur = df[df["outcome"] == 1]["prglngth"]
preg_dur.value_counts()

# %%
fig, ax = plt.subplots()
ax.hist(preg_dur, bins=np.arange(20,50), edgecolor="black", color="skyblue")
plt.show()

# %% [markdown]
# By far the most common quantity is 39 weeks. Which is roughly 9 months. Well, this is expected too.

# %% [markdown]
# Early babies are common, but doctors often intervene if it goes past 43 weeks.

# %% [markdown]
# ## Outliers

# %% [markdown]
# Let's look at the outliers of these distributions.

# %%
preg_dur.value_counts().sort_index(ascending=True)

# %% [markdown]
# - So looking at this frequency table, we can say that pregnancy lengths less than 10 weeks are probably errors since we are only evaluating live births.
# - Lengths higher than 30 are probably legitimate.
# - Between 10 and 30 weeks, it is hard to be sure. Some of the quantities are probably errors, but some are correctly recorded preterm babies.

# %%
preg_dur.value_counts().sort_index(ascending=False)

# %% [markdown]
# We can interpert this outliers further, but best way to handle them is having "domain knowledge".

# %% [markdown]
# ## First Babies

# %% [markdown]
# Let's compare the distribution of pregnancy lengths for first babies and others.

# %%
first = df[(df["outcome"] == 1) & (df["birthord"] == 1)]["prglngth"]
others = df[(df["outcome"] == 1) & (df["birthord"] != 1)]["prglngth"]

# %%
fig, ax = plt.subplots()
ax.hist([first, others], bins=np.arange(25,47))
ax.set_ylabel("Frequency")
ax.set_xlabel("Pregnancy length (weeks)")
plt.show()

# %% [markdown]
# This roughly shows us that there are more non-first babies born during week 39. But we shouldn't compare them directly.
#
# Let's just comapre the means.

# %%
print(f"First {first.mean()}")
print(f"Others {others.mean()}")

# %% [markdown]
# Difference is only 13 hours. We can't really say that the difference is significant.

# %% [markdown]
# There are several possible causes for this difference:
# - There is an actual difference in pregnancy length between first babies and other babies.
# - It is caused by some kind of bias in the sampling process.
# - It is a measurement error.
# - It is casued by random variation in the sampling process.

# %% [markdown]
# ## Effect Size

# %% [markdown]
# Comparing the means is only effective if we are aware of the unit we use.
#
# But to get e unit-less statistic, we can standardize the effect size. _Cohen's d_ is the most popular way of doing this:

# %% [markdown]
# $$
# d = \frac{\bar{x}_1 - \bar{x}_2}{s}
# $$

# %% [markdown]
# We can't use the standard deviation of one of the groups blindly. So we calculate the pooled standard deviation here with the below formula, basically finding the average standard deviation (for `ddof=1`):
#
# $$
# s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}
# $$

# %%
var1, var2 = first.var(), others.var()
print(f"Variance (s^2) of the first babies: {var1}")
print(f"Variance (s^2) of the other babies: {var2}")

# %%
n1, n2 = first.count(), others.count()
print(f"Number of first babies: {n1}")
print(f"Number of other babies: {n2}")

# %%
from math import sqrt

s_pooled = sqrt(
    ((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2)
)
print(f"s_pooled: {s_pooled}")

# %% [markdown]
# Now, we can find the differnece in the form of standard deviation.

# %%
diff = first.mean() - others.mean()
cohen_d = diff / s_pooled
cohen_d

# %% [markdown]
# Difference is very small:
# - **0.2** = small effect
# - **0.5** = moderate effect
# - **0.8** = large effect

# %% [markdown]
# ## Exercises

# %%
from statadict import parse_stata_dict
fem_dict = parse_stata_dict('./data/2002FemResp.dct')
fem = pd.read_fwf('./data/2002FemResp.dat', names=fem_dict.names, colspecs=fem_dict.colspecs)

# %%
totinc = fem["totincr"]
fig, ax = plt.subplots()
ax.hist(totinc, color="skyblue", edgecolor="black", bins=np.arange(1,15))
plt.show()

# %%
parity = fem["parity"]
fig, ax = plt.subplots()
ax.hist(parity, color="skyblue", edgecolor="black", bins=np.arange(1,10))
plt.show()

# %%
parity.value_counts().sort_index(ascending=False)

# %% [markdown]
# Maybe the woman with 22 childeren is an input error? Or is it just a rare case?

# %%
low_income_parity=fem[fem["totincr"].between(1,6, inclusive="both")]["parity"]
high_income_parity=fem[fem["totincr"].between(13,14, inclusive="both")]["parity"]

# %%
s_pooled = sqrt(
    ((low_income_parity.count() - 1) * low_income_parity.var() + (high_income_parity.count() - 1) * high_income_parity.var()) /
    (low_income_parity.count() + high_income_parity.count() - 2)
)

print(low_income_parity.std())
print(high_income_parity.std())
print(s_pooled)
print(f"Cohen's d for low income/high income difference: {(low_income_parity.mean() - high_income_parity.mean())/s_pooled}")

# %% [markdown]
# We see that the effect of income on parity is small to moderate (in 2002!). Let's also see it on the histogram.

# %%
fig, axes = plt.subplots(1,2, figsize=(10,4))

axes[0].hist(low_income_parity, color="green",edgecolor="black", bins=np.arange(0,15))
axes[1].hist(high_income_parity, color="red",edgecolor="black",bins=np.arange(0,15))
axes[0].set_title("Low Income")
axes[1].set_title("High Income")

for ax in axes:
    ax.set_xlabel("Parity (number of children)")
    ax.set_ylabel("Frequency")
