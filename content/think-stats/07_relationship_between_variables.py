# %% [markdown]
r"""
---
title: "Relationships Between Variables"
date: 2026-05-25
tags: [statistics, think-stats, correlation, scatter, python]
description: "Scatter plots, Pearson correlation, Spearman rank correlation, and covariance."
---
"""


# %% [markdown]
r"""
# Relationship between variables

There is a popular survey called NLSY97, it has data about students taking verbal and mathematical tests, we are going to use this data for seeing that if a student good at math is also good at verbal exams too.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import spearmanr

# %%
stud = pd.read_csv("./data/nlsy97-extract.csv.gz").replace([-1, -2, -3, -4, -5], np.nan)

# %%
stud["sat_verbal"] = stud["R9793800"]
stud["sat_math"] = stud["R9793900"]

# %%
cols = ["sat_verbal", "sat_math"]
for c in cols:
    invalid = stud[c] < 200
    stud.loc[invalid, c] = np.nan

stud_valid = stud.dropna(subset=cols).copy()
stud_valid.shape

# %%
stud_valid[["sat_verbal", "sat_math"]]

# %% [markdown]
r"""
## Scatter Plots

Now that we have our columns, let's see if there is a correlation between verbal and math score.
"""

# %%
fig, ax = plt.subplots()
ax.scatter(stud_valid["sat_verbal"], stud_valid["sat_math"], s=5)
plt.show()

# %% [markdown]
r"""
Becuase the scores are rounded of the nearest multiple of 10, we cannot bring back the data, but we can add random noise.
"""


# %%
def jit(seq, std=1):
    return np.random.normal(0, std, len(seq)) + seq


# %%
sat_verbal_jitter = jit(stud_valid["sat_verbal"], 3)
sat_math_jitter = jit(stud_valid["sat_math"], 3)

# %%
fig, ax = plt.subplots()
ax.scatter(sat_verbal_jitter, sat_math_jitter, s=5, alpha=0.5)
ax.set_xlabel("SAT Math")
ax.set_ylabel("SAT Verbal")
plt.show()

# %% [markdown]
r"""
We can see some kind of correlation, but we had to tweak a lot of stuff to get the visualization right.
"""

# %% [markdown]
r"""
## Decile Plots

We have to divide the data to deciles, hence the name of the plot.
"""

# %%
deciles = pd.qcut(stud_valid["sat_verbal"], 30, labels=False) + 1
deciles.value_counts().sort_index()

# %%
stud_groupby = stud_valid.groupby(deciles)
math_groupby_ser = stud_groupby["sat_math"]
median = math_groupby_ser.quantile(0.5)
low = math_groupby_ser.quantile(0.1)
high = math_groupby_ser.quantile(0.9)
ixs = stud_groupby["sat_verbal"].quantile(0.5)
ixs

# %%
fig, ax = plt.subplots()
ax.fill_between(ixs, low, high, alpha=0.2)
ax.plot(ixs, median, label="median")
ax.set_xlabel("SAT Verbal")
ax.set_ylabel("SAT Math")
plt.show()

# %% [markdown]
r"""
We can see that relationship between SAT Math score and SAT Verbal score of respondents are somewhat has linear correlation.
"""

# %% [markdown]
r"""
## Correlation

When NLSY respondents were in 9th grade, many of them took the math section of the PIAT. Let's see if there is a correlation between PIAT Math score and the SAT Math score.
"""

# %%
stud_valid["piat_math"] = stud_valid["R1318200"]
stud["piat_math"] = stud["R1318200"]
stud_piat_sat = stud_valid.dropna(subset=["piat_math"])
stud_piat_sat.reset_index(inplace=True)
stud_piat_sat

# %%
stud_piat_sat_pdf = (
    stud_piat_sat["piat_math"].value_counts(normalize=True).sort_index().cumsum()
)

# %%
fig, ax = plt.subplots()
ax.plot(stud_piat_sat_pdf.index, stud_piat_sat_pdf)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(stud_piat_sat["piat_math"], stud_piat_sat["sat_math"], s=5)
ax.set_xlabel("PIAT Math")
ax.set_ylabel("SAT Math")
plt.show()

# %% [markdown]
r"""
I just want to see if visualization would get better if I used min-max scaler for both columns
"""

# %%
scaler = MinMaxScaler()
stud_piat_sat["piat_math_minmax"] = scaler.fit_transform(stud_piat_sat[["piat_math"]])
stud_piat_sat["sat_math_minmax"] = scaler.fit_transform(stud_piat_sat[["sat_math"]])
stud_piat_sat

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(stud_piat_sat["piat_math_minmax"], stud_piat_sat["sat_math_minmax"], s=5)
ax.set_xlabel("PIAT Math (min-max scaled)")
ax.set_ylabel("SAT Math (min-max scaled)")
plt.show()

# %% [markdown]
r"""
Again, we can rougly say that people who did well in PIAT Math are likely to do well on SAT Math.
"""

# %% [markdown]
r"""
But since the title is "correlation", let's investigate **Pearson correlation coefficient**, often just called **correlation**.

Before that we have to standardize both of the columns to standardize. Standard score of a sample X can be calculate with the below formula.

$$
Z = \frac{x - \bar{x}}{s}
$$

But we are going to use `StandardScaler` which also does the same thing for a dataset.
"""

# %%
std_scaler = StandardScaler()
stud_piat_sat["piat_math_norm"] = std_scaler.fit_transform(stud_piat_sat[["piat_math"]])
stud_piat_sat["sat_math_norm"] = std_scaler.fit_transform(stud_piat_sat[["sat_math"]])
stud_piat_sat[["piat_math_norm", "sat_math_norm"]]

# %%
fig, axes = plt.subplots(2, 1, figsize=(7, 4))
axes[0].axhline(y=0, color="black", alpha=0.15)
axes[1].axhline(y=0, color="black", alpha=0.15)
axes[0].plot(stud_piat_sat["piat_math_norm"])
axes[1].plot(stud_piat_sat["sat_math_norm"])
axes[0].set_xlim(100, 200)
axes[1].set_xlim(100, 200)
axes[0].set_ylabel("z-score")
axes[1].set_ylabel("z-score")
plt.show()

# %% [markdown]
r"""
Even though it's hard to read it like this, we can totally see some kind of correlation here.
"""

# %% [markdown]
r"""
We can find the correlation coefficients using the below formula, and since we calculated the z-scores of our relevant columns, There is just one step left.

$$
r = \frac{\sum (z_x z_y)}{n-1}
$$
"""

# %%
print("r =", np.mean(stud_piat_sat["piat_math_norm"] * stud_piat_sat["sat_math_norm"]))

# %% [markdown]
r"""
This means that, if someone's PIAT Math score is 1 standard deviation above the mean, we expect their SAT math score to be 0.64 standard deviations above the mean, on average. Since the correlation coefficient is symmetric, the opposite is also true.
"""

# %%
print("r =", np.mean(stud_piat_sat["sat_math_norm"] * stud_piat_sat["piat_math_norm"]))

# %% [markdown]
r"""
We can also use the relevant `numpy` function to find the correlation.
"""

# %%
print("r =", np.corrcoef(stud_piat_sat["piat_math"], stud_piat_sat["sat_math"]))

# %% [markdown]
r"""
There are also non-linear correlations too, so we cannot say that there is no correlation if the correlation coefficient is zere. In that case, we may have to investigate further.
"""

# %% [markdown]
r"""
## Rank Correlation
"""

# %% [markdown]
r"""
NLSY dataset also has relatively recent data about the incomes of the respondents.
"""

# %%
stud_valid["income"] = stud_valid["U4949700"]
stud["income"] = stud["U4949700"]
stud["income"].describe()

# %%
income_pmf = stud["income"].value_counts(normalize=True).sort_index()
income_cdf = income_pmf.cumsum()

# %%
fig, ax = plt.subplots()
ax.plot(income_cdf.index, income_cdf)
plt.show()

# %% [markdown]
r"""
Let's look at the correlation between SAT Math scores and incomes.
"""

# %%
fig, ax = plt.subplots()
ax.scatter(stud["piat_math"], stud["income"], s=2)
plt.show()

# %%
stud["piat_math"].corr(stud["income"])

# %% [markdown]
r"""
This is not a strong correlation as the correlation between SAT scores and PIAT scores. But considering the number of factors that affect income, it's still pretty strong.
"""

# %% [markdown]
r"""
Correlation coefficient is affected by outliers a lot, so let's user something better for this scenario.
"""

# %%
valid = stud.dropna(subset=["piat_math", "sat_math", "income"])

# %%
piat_math_rank = valid["piat_math"].rank(method="first")
income_rank = valid["income"].rank(method="first")

# %%
fig, ax = plt.subplots()
ax.scatter(piat_math_rank, income_rank, s=1, alpha=0.6)
plt.show()

# %%
np.corrcoef(piat_math_rank, income_rank)

# %% [markdown]
r"""
It's still not a very strong correlation, but it is better than using correlation coefficient on its own, since we have a lot of outilers. We can use `scipy`'s `spearmanr` method, which does the same.
"""

# %%
spearmanr(valid["income"], valid["piat_math"]).statistic

# %% [markdown]
r"""
Let's do the same with SAT scores too.
"""

# %%
valid["sat_math"].corr(valid["income"])

# %%
fig, ax = plt.subplots()
ax.scatter(valid["sat_math"], valid["income"], s=3)
plt.show()

# %%
spearmanr(valid["sat_math"], valid["income"]).statistic

# %% [markdown]
r"""
It's still not a strong correlation, but strong considering the income factors.
"""

# %% [markdown]
r"""
## Correlation and Causation

[Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation). Identifying and measuring causal relationships is the topic of a branch of statistics called causal inference.
"""

# %% [markdown]
r"""
## Exercises

Let's investigate the relationship between degrees and income.
"""

# %%
stud["degree"] = stud["Z9083900"]
deginc = stud.dropna(subset=["degree", "income"])

# %%
deginc_grouped = deginc.groupby("degree")
deginc_grouped

# %%
deginc_grouped_medians = deginc_grouped.quantile(0.5)
deginc_grouped_medians["income"]

# %%
fig, ax = plt.subplots()
ax.plot(deginc_grouped_medians["income"].index, deginc_grouped_medians["income"])
ax.set_xticks(
    ticks=deginc_grouped_medians["income"].index,
    labels=[
        "None",
        "GED",
        "High school diploma",
        "Associate's degree",
        "Bachelor's degree",
        "Master's degree",
        "PhD",
        "Professional degree",
    ],
    ha="right",
    rotation=30,
)
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
## Glossary from the resource 

- **scatter plot:** A visualization that shows the relationship between two variables by plotting one point for each observation in the dataset.
- **overplotted:** A scatter plot is overplotted if many markers overlap, making it hard to distinguish areas of different density, which can misrepresent the relationship.
- **jitter:** Random noise added to data points in a plot to make overlapping values more visible.
- **decile plot:** A plot that divides data into deciles (ten groups) based on one variable, then summarizes another variable for each group.
- **decile:** One of the groups created by sorting data and dividing it into ten roughly equal parts.
- **Pearson correlation coefficient:** A statistic that measures the strength and sign (positive or negative) of the linear relationship between two variables.
- **standard score:** A quantity that has been standardized so that it is expressed in standard deviations from the mean.
- **correlation matrix:** A table showing the correlation coefficients for each pair of variables in a dataset.
- **rank correlation:** A robust way to quantify the strength of a relationship by using the ranks of values instead of the actual values.
- **randomized controlled trial:** An experiment where subjects are randomly assigned to groups that receive different treatments.
- **treatment group:** In an experiment, the group that receives the intervention being tested.
- **control group:** In an experiment, the group that does not receive the intervention, or receives a treatment whose effect is known.
- **natural experiment:** An experiment that uses naturally occurring groups, which can sometimes mimic random assignment.
- **causal inference:** Methods for identifying and quantifying cause-and-effect relationships.
"""
