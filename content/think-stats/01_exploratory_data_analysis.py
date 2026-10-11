# %% [markdown]
r"""
---
title: "Exploratory Data Analysis"
date: 2026-05-25
tags: [statistics, think-stats, eda, python]
description: "Introduction to exploratory data analysis, data cleaning, and summary statistics."
---
"""


# %% [markdown]
# # Exploratory Data Analysis

# %%
import numpy as np
import pandas as pd
from scipy.constants import pound as POUND

# %%
from statadict import parse_stata_dict

# %%
stata_dict = parse_stata_dict('./data/2002FemPreg.dct')
resp = pd.read_fwf('./data/2002FemPreg.dat', names=stata_dict.names, colspecs=stata_dict.colspecs)
resp

# %% [markdown]
# I just want to check if data is corrupted since a lot of rows have `NaN` for howpreg_n

# %%
resp[resp["howpreg_n"].notna()]

# %% [markdown]
# No, data is not corrupted. Checking the [codebook](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NSFG/Cycle6Codebook-Pregnancy.pdf) may help for better resolution.

# %% [markdown]
# Now, let's check the shape of this dataframe.

# %%
resp.shape

# %% [markdown]
# It has 243 variables about 13593 pregnancies.

# %% [markdown]
# Let's check the `outcome` variable.

# %%
resp["outcome"].value_counts().sort_index()

# %% [markdown]
# `value_counts()` counts the number of different values, useful for counting elements for different categorical variables.

# %%
for i in resp.columns:
    if i.startswith("birth"):
        print(i)

# %% [markdown]
# Also, here are the birth weights.

# %%
resp["birthwgt_lb"].value_counts(dropna=False).sort_index()

# %% [markdown]
# As we can see, there are weights far above what a baby could possibly weigh. We are going to replace those values with `NaN`.

# %%
resp["birthwgt_lb"] = resp["birthwgt_lb"].replace([15,51,97,98,99], np.nan)

# %% [markdown]
# Now it is clearer.

# %%
resp["birthwgt_lb"].value_counts(dropna=False).sort_index()

# %% [markdown]
# Now, let's look at the average age of women for the end of birth. 

# %%
for i in resp.columns:
    if i.startswith("age"):
        print(i)

# %%
resp["agepreg"].mean()

# %% [markdown]
# In this dataset, `agepreg` is stored as centiyear (which is hundredths of a year), it would be eaiser for us if we transformed this column into years.

# %%
resp["agepreg"] /= 100

# %%
resp["agepreg"]

# %% [markdown]
# Now, let's look at the average of `agepreg`. Mother's age at the end of the pregnancy.

# %%
resp["agepreg"].mean()

# %%
resp[['birthwgt_lb', 'birthwgt_oz']]

# %% [markdown]
# Coming back to weights, we see that birth weight is separated to pounds and ounces, we will combine them in `totalwgt_lb`.

# %%
resp["totalwgt_lb"] = resp["birthwgt_lb"] + resp["birthwgt_oz"] / 16.0
resp[['totalwgt_lb','birthwgt_lb', 'birthwgt_oz']]

# %% [markdown]
# Also, who uses imperal system anyway? Let's convert this to kilograms.

# %%
resp["totalwgt_kg"] = resp["totalwgt_lb"] * POUND
resp["totalwgt_kg"]

# %% [markdown]
# We are getting "highly fragmented dataframe" warning, let's fix it.

# %%
resp = resp.copy()
resp

# %% [markdown]
# For now, we fixed it. But using dictionaries, creating a data frame with it, then concatenating them in one of those data frames is much more efficient when there are innumerous amount of variables.

# %% [markdown]
# Now we can summarize the statistics.

# %%
weights = resp["totalwgt_kg"]
n = weights.count()
weights

# %%
print(weights.sum() / n)
print(weights.mean())
mean = weights.mean()

# %% [markdown]
# $$
# SS = \sum (x_i - \hat{x})^2
# $$

# %%
squared_deviations = (weights - mean) ** 2
squared_deviations

# %% [markdown]
# $$
# s^2 = \frac{SS}{n-1}
# $$

# %%
print(squared_deviations.sum() / (n - 1))
var = weights.var(ddof=1)

# %% [markdown]
# $$
# s = \sqrt{\frac{SS}{n-1}}
# $$

# %%
print(np.sqrt(var))
std = weights.std(ddof=1)
print(std)

# %% [markdown]
# Informally, values that are one or two SDs from the mean are common.
#
# There are number of ways to interpret this data. But for this exercise, we are going to analyse `caseid` 10229 

# %%
subset = resp[resp["caseid"] == 10229]
subset["outcome"]

# %% [markdown]
# `1` indicates a live birth, `4` indicates a miscarriage. If we consider this data with empathy, it is natural to be moved by the story it tells.

# %% [markdown]
# Finally, let's do some practice.

# %%
resp["birthord"].value_counts().sort_index()

# %%
resp[resp["caseid"] == 2298]

# %%
resp[(resp["caseid"] == 5013) & (resp["pregordr"] == 1)]["totalwgt_kg"]

# %%
resp.to_csv("./data/2002FemPreg_after_01.csv", index=False)

# %% [markdown]
# ## Glossary from the resource
#
# - **anecdotal evidence:** Data collected informally from a small number of individual cases, often without systematic sampling.
# - **cross-sectional study:** A study that collects data from a representative sample of a population at a single point or interval in time.
# - **cycle:** One data-collection interval in a study that collects data at multiple intervals in time.
# - **population:** The entire group of individuals or items that is the subject of a study.
# - **sample:** A subset of a population, often chosen at random.
# - **respondents:** People who participate in a survey and respond to questions.
# - **representative:** A sample is representative if it is similar to the population in ways that are important for the purposes of the study.
# - **stratified:** A sample is stratified if it deliberately oversamples some groups, usually to make sure that enough members are included to support valid conclusions.
# - **oversampled:** A group is oversampled if its members have a higher chance of appearing in a sample.
# - **variable:** In survey data, a variable is a collection of responses to questions or values computed from responses.
# - **codebook:** A document that describes the variables in a dataset, and provides other information about the data.
# - **recode:** A variable that is computed based on other variables in a dataset.
# - **raw data:** Data that has not been processed after collection.
# - **data cleaning:** A process for identifying and correcting errors in a dataset, dealing with missing values, and computing recodes.
# - **statistic:** A value that describes or summarizes a property of a sample.
# - **standard deviation:** A statistic that quantifies the spread of data around the mean.
