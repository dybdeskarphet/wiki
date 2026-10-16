# %% [markdown]
r"""
---
title: "Covariance, Correlation, and OLS"
date: 2026-06-22
tags: [statistics, linear-algebra, regression, ols, covariance]
description: "Connecting covariance, correlation coefficient, and ordinary least squares regression from first principles."
---
"""


# %% [markdown]
r"""
## Covariance
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# %% [markdown]
r"""
First, we define two random variables with sample size ($n$) 500.

$$
\begin{gathered}
X \sim Exp(\lambda=0.5) \\\\
\epsilon \sim \mathcal{N}(\mu=9, \sigma^2=4) \\\\
Y \sim 3X + \epsilon
\end{gathered}
$$
"""

# %%
X = np.random.exponential(scale=2.0, size=500)
Y = 3.0 * X + np.random.normal(loc=9.0, scale=2.0, size=500)

# %%
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# %%
print(f"E[X] = {mean_X}")
print(f"E[Y] = {mean_Y}")

# %% [markdown]
r"""
Since we are interested in calculating the covariance, let’s calculate the deviations for both X and Y first.

$$
\begin{gathered}
\text{dev_X} = X_i - \bar{X} \\\\
\text{dev_Y} = Y_i - \bar{Y}
\end{gathered}
$$
"""

# %%
dev_X = X - mean_X
dev_X

# %%
dev_Y = Y - mean_Y
dev_Y

# %% [markdown]
r"""
To find the covariance of X and Y, we calculate the mean of the product of the deviations of the two random variables that is the expected value.

$$
Cov(X,Y) = E[\text{dev_X}\cdot\text{dev_Y}]
$$
"""

# %%
cov_XY = np.mean(dev_X * dev_Y)
cov_XY

# %%
cov_matrix = np.cov(X, Y, bias=True)
cov_matrix

# %% [markdown]
r"""
As we can see below, NumPy’s built-in method for calculating covariance also yields the same result as the one we derived.
"""

# %%
np.isclose(cov_matrix[0][0], np.mean(dev_X * dev_X))

# %% [markdown]
r"""
## Correlation

The only difference between correlation and covariance is that correlation is normalized.

$$
\begin{gathered}
z_X = \frac{X_i - E[X]}{\sigma_X} \\\\
z_Y = \frac{Y_i - E[Y]}{\sigma_Y}
\end{gathered}
$$
"""

# %%
z_X = dev_X / np.std(X, ddof=1)
z_Y = dev_Y / np.std(Y, ddof=1)

# %% [markdown]
r"""
While in covariance we directly calculate the mean of the deviations, here we calculate the mean of the Z-scores, which is actually the mean of the standardized deviations.

$$
\rho_{X,Y} = E[z_X z_Y]
$$
"""

# %%
corr_XY = np.mean(z_X * z_Y)
corr_XY

# %% [markdown]
r"""
As we can see, NumPy’s built-in function also returns very similar values.
"""

# %%
print(np.corrcoef(X, Y)[0][1], corr_XY)

# %% [markdown]
r"""
## Ordinary Least Squares

When creating an OLS model, we actually build it entirely using covariances and variances. Let's remember the model first:

$$
Y = \beta_0 + \beta_1 X
$$

First, let’s define $\beta_1$:

$$
\beta_1 = \frac{Cov(X,Y)}{Var(X)}
$$

The result of this calculation will tell us how much Y changes for every unit change in X. This makes perfect sense when we consider that covariance measures the “rate of joint change.”
"""

# %%
slope = cov_XY / np.var(X)
slope

# %% [markdown]
r"""
However, to determine the y-intercept, it is not enough to know the slope; we also need to know the value of Y when X is 0.

To do this, we first swap the variables in our model function: 

$$
\beta_0 = Y - \beta_1 X
$$

Then we substitute $\bar{X}$ and $\bar{Y}$ for X and Y. 
"""

# %%
intercept = mean_Y - (slope * mean_X)
intercept

# %%
model = linregress(X, Y)
model.__getattribute__("slope")

# %% [markdown]
r"""
We can also see that the values we actually found are correct when we check them using the `linregress` function from `scipy.stats`.
"""

# %%
print(
    "Is manuall found intercept and scipy's interecept close?",
    np.isclose(intercept, model.__getattribute__("intercept")),
)
print(
    "Is manuall found slope and scipy's slope close?",
    np.isclose(slope, model.__getattribute__("slope")),
)

# %% [markdown]
r"""
Our goal in linear regression was to minimize the sum of squared errors; below, we can see the sum of squared errors we obtained.
"""

# %%
Y_pred = intercept + (slope * X)
residuals = Y - Y_pred
print("Sum of squared errors:", np.sum(residuals**2))

# %% [markdown]
r"""
Here, too, we can see how our data points are distributed in the scatter plot. The line passing through them is the one that minimizes the sum of the squared errors, in other words, our linear regression line.
"""

# %%
fig, ax = plt.subplots()
ax.scatter(X, Y, alpha=0.1, marker=".", color="red")
ax.plot(X, Y_pred, label="model")
ax.axvline(
    mean_X, ymax=mean_Y, linestyle="--", alpha=0.5, label=f"$\\bar{{X}}$: {mean_X:.2f}"
)
ax.axhline(
    mean_Y, xmax=mean_X, linestyle="--", alpha=0.5, label=f"$\\bar{{Y}}$: {mean_Y:.2f}"
)
ax.legend()
plt.show()

# %% [markdown]
r"""
From our graph, we can also see that we have aligned the point where the line intersects the Y-axis ($\beta_0$) with the point where the line passes through the means of both X and Y.
"""

# %%
np.isclose((intercept + (slope * mean_X)), mean_Y)
