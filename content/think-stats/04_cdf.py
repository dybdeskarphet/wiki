# %% [markdown]
r"""
---
title: "Cumulative Distribution Functions (CDF)"
date: 2026-05-25
tags: [statistics, think-stats, probability, cdf, python]
description: "CDFs, percentiles, and comparing empirical distributions."
---
"""


# %% [markdown]
# # CDF (Cumulative Distribution Functions)
#
# ## NSFG Data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
preg = pd.read_csv("./data/2002FemPreg_after_01.csv")
preg

# %%
firsts, others = preg[(preg["pregordr"] == 1) & (preg["outcome"] == 1)], preg[(preg["pregordr"] != 1) & (preg["outcome"] == 1)]

# %%
first_weights = firsts["totalwgt_kg"].dropna()
first_weights.mean()

# %%
others_weights = others["totalwgt_kg"].dropna()
others_weights.mean()

# %% [markdown]
# We see that in general, non-first children are much heavier than the first children. Let's investigate this, because using only the mean to make this assumption is risky.
#
# Let's look at the PMFs first.

# %%
first_freq = first_weights.value_counts(normalize=True).sort_index()
first_freq

# %%
other_freq = others_weights.value_counts(normalize=True).sort_index()
other_freq

# %%
width = 0.02
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(first_freq.index + width / 2, first_freq, width=width, label="firsts", alpha=0.4)
ax.bar(other_freq.index - width / 2, other_freq, width=width, label="others", alpha=0.4)
ax.legend()
plt.show()

# %% [markdown]
# This is hard to look at. No conclusions, I just wanted to say it. We could bin them, but useful information would be gone too with the bins. 

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(first_weights,bins=30, density=True, alpha=0.4)
ax.hist(others_weights,bins=30, density=True, alpha=0.4)
plt.show()

# %% [markdown]
# Okay it looks better, but as i said, we surely lost information while binning. And it is still hard to interpolate.

# %%
first_cdf = first_freq.cumsum()
others_cdf = other_freq.cumsum()

fig, ax = plt.subplots(figsize=(10,6))
ax.step(first_cdf.index, first_cdf, label="firsts")
ax.step(others_cdf.index, others_cdf, label="others")
ax.legend()
plt.show()

# %% [markdown]
# Curve of the first babies are to the left. Which means that first babies are slightly lighter through out the distribution.

# %% [markdown]
# ## Percentile-Based Statistics

# %% [markdown]
# Let's investigate percentile based statistics, which are less sensitive to outliers.

# %%
from statadict import parse_stata_dict

# %%
stata_dict = parse_stata_dict('./data/2002FemPreg.dct')
preg = pd.read_fwf('./data/2002FemPreg.dat', names=stata_dict.names, colspecs=stata_dict.colspecs)
preg

# %%
lb = preg["birthwgt_lb"]
oz = preg["birthwgt_oz"]

# %% [markdown]
# 1 pound is 16 ounces, we assume that above values indicate missing data

# %%
oz_freq = oz.value_counts().sort_index()
oz_freq

# %% [markdown]
# Now, we can replace these invalid values. However, assuming that they will mostly go unnoticed, let's see how our statistics change.

# %%
lb_clean= lb.replace([51,97,98,99], np.nan)
oz_clean= oz.replace([97,98,99], np.nan)

# %%
total_weight_clean = lb_clean + (oz_clean / 16)
total_weight_invalid = lb + (oz / 16)
total_weight_clean = total_weight_clean.dropna()

# %% [markdown]
# Let's compute the mean and see the difference.

# %%
print(total_weight_clean.mean())
print(total_weight_invalid.mean())

# %% [markdown]
# Difference is very small, but remember, mean is only the first moment of this dataset. Let's compute the standard deviation.

# %%
print(total_weight_clean.std())
print(total_weight_invalid.std())

# %% [markdown]
# Difference is getting larger. Let's compute the skewness which we learned eariler.

# %%
print(total_weight_clean.skew())
print(total_weight_invalid.skew())

# %% [markdown]
# If we only wanted to find the skewness, _invalid_ dataset would give us a completely wrong statistic, even the sign of the skewnes is wrong.

# %% [markdown]
# To start calculating with percentile-based statistics, we have to create a CDF first.

# %%
total_cdf = total_weight_invalid.value_counts(normalize=True).sort_index().cumsum()
total_cdf

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax.step(total_cdf.index, total_cdf, label="firsts")

# %%
total_cdf_clean= total_weight_clean.value_counts(normalize=True).sort_index().cumsum()
total_cdf_clean

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax.step(total_cdf_clean.index, total_cdf_clean, label="firsts")

# %% [markdown]
# Median is a percentile-based statistic. Let's compare the clean and invalid datasets.

# %%
total_weight_clean.median(), total_weight_invalid.median()

# %% [markdown]
# As you can see, median is much less sensitive to outliers. Let's find the IQR, the difference between 75th and 25th percentile.

# %%
total_iqr = total_cdf[total_cdf >= 0.75].index[0] - total_cdf[total_cdf >= 0.25].index[0]
total_iqr_clean = total_cdf_clean[total_cdf_clean >= 0.75].index[0] - total_cdf_clean[total_cdf_clean >= 0.25].index[0]
total_iqr, total_iqr_clean

# %% [markdown]
# In general, outliers have less effect on the IQR than the standard deviation.

# %% [markdown]
# We may ask, why are we comparing IQR with standard deviation:
#
# We can categorize statistical calculations into two groups: moment-based statistics and percentile-based statistics. The mean is moment-based, while the median is percentile-based. The standard deviation is moment-based, while the IQR is percentile-based. There are also very handy calculations we can use if the distribution is Gaussian, but we will not delve into this topic for now.

# %% [markdown]
# We remember that skewness was very off using moment-based statistics. To find the percentile-based skewness, we use Bowley's Skewness formula.
#
# $$
# Skew = \frac{(Q_3 - Q_2) - (Q_2 - Q_1)}{Q_3 - Q_1}
# $$

# %%
q1, q2, q3 = total_weight_invalid.quantile([0.25,0.5,0.75])
# c symbolizng clean
q1_c, q2_c, q3_c = total_weight_clean.quantile([0.25,0.5,0.75])

skew = ((q3-q2) - (q2-q1)) / (q3-q1)
skew_clean = ((q3_c-q2_c) - (q2_c-q1_c)) / (q3_c-q1_c)
skew, skew_clean

# %% [markdown]
# ## Exercises

# %%
preg = pd.read_csv("./data/2002FemPreg_after_01.csv")
preg

# %%
male_wgt = preg[preg["babysex"] == 1]["totalwgt_kg"]
female_wgt = preg[preg["babysex"] == 2]["totalwgt_kg"]

# %% [markdown]
# Let's see the difference of birth weights between male and female babies. Using CDF.

# %%
male_pdf = male_wgt.value_counts(normalize=True).sort_index()
female_pdf = female_wgt.value_counts(normalize=True).sort_index()
male_cdf = male_pdf.cumsum()
female_cdf = female_pdf.cumsum()
male_cdf, female_cdf

# %%
fig, axes = plt.subplots(1,2,figsize=(16,6))
axes[0].step(male_cdf.index, male_cdf, label="male")
axes[0].step(female_cdf.index, female_cdf, label="female")
axes[1].fill_between(male_pdf.index, male_pdf, label="male", alpha=0.4)
axes[1].fill_between(female_pdf.index, female_pdf, label="female",alpha=0.4)
axes[0].legend()
axes[1].legend()
plt.show()

# %% [markdown]
# We can say that female babies are slightly lighter than male babies. We can also assume that if the CDF plot looks like an S curve, distribution of the data is likely a bell-curve (we cannot specifically assume that if it's Gaussian or Logistic or whatever). 

# %% [markdown]
# Let's do a final exercise. Normally, `np.random.random` gives us a uniform distribution. We can test this by creating a CDF from it.

# %%
t = np.random.random(1001)
t_freq = pd.Series(t).value_counts(normalize=True).sort_index().cumsum()
fig, ax = plt.subplots()
ax.step(t_freq.index, t_freq)
plt.show()

# %% [markdown]
# It is in fact, a uniform distribution.

# %% [markdown]
# ## Glossary from the resource
#
# - **percentile rank:** The percentage of values in a distribution that are less than or equal to a given quantity.
# - **percentile:** The value in a distribution associated with a given percentile rank.
# - **cumulative distribution function (CDF):** A function that maps a value to the proportion of the distribution less than or equal to that value.
# - **quantile:** The value in a distribution that is greater than or equal to a given proportion of values.
# - **robust:** A statistic is robust if it is less affected by extreme values or outliers.
# - **interquartile range (IQR):** The difference between the 75th and 25th percentiles, used to measure the spread of a distribution.
