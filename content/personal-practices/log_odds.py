# %% [markdown]
r"""
---
title: "Understanding Log-odds"
date: 2026-10-20
tags: [log-odds, logarithmic, likelihood, fisher-information]
description: ""
---
"""

# %% [markdown]
r"""
## Likelihood Function
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import simpson

# %%
TRUE_P = 0.7

# %%
data = np.random.binomial(n=1, p=TRUE_P, size=500)
log_likelihood_reason = np.random.binomial(n=1, p=TRUE_P, size=5000)
data[:10]

# %%
def raw_likelihood(p, data):
    ones = np.sum(data == 1)
    zeros = np.sum(data == 0)
    return (p ** ones) * ((1-p) ** zeros)

# %%
likelihoods = []
likelihoods_comparison = []
for p in np.linspace(start=0,stop=1,num=20):
    l = raw_likelihood(p,data)
    l_comp = raw_likelihood(p,log_likelihood_reason)
    print(f"If p={p}, Likelihood is:", l)
    likelihoods.append((p,l))
    likelihoods_comparison.append((p,l_comp))

# %%
fig, axes = plt.subplots(1,2, figsize=(12,4))
x,y = zip(*likelihoods)
x_comp,y_comp = zip(*likelihoods_comparison)
axes[0].plot(x,y)
axes[1].plot(x_comp,y_comp)
plt.show()

# %% [markdown]
r"""
## Log-likelihood Function
"""

# %%
def log_likelihood(p,data):
    ones = np.sum(data == 1)
    zeros = np.sum(data == 0)
    return ones * np.log(p) + zeros * np.log(1-p)


# %%
log_likelihoods = []
log_likelihoods_comparison = []
for p in np.linspace(start=0,stop=1,num=20):
    l = log_likelihood(p,data)
    l_comp = log_likelihood(p,log_likelihood_reason)
    print(f"If p={p}, Log-likelihood is:", l)
    log_likelihoods.append((p,l))
    log_likelihoods_comparison.append((p,l_comp))

# %%
fig, axes = plt.subplots(1,2, figsize=(12,4))
x,y = zip(*log_likelihoods)
x_comp,y_comp = zip(*log_likelihoods_comparison)
axes[0].plot(x,y)
axes[1].plot(x_comp,y_comp)
plt.show()

# %% [markdown]
r"""
## Fisher Information
"""

# %%
n = 500
sample = np.random.binomial(n=1, p=TRUE_P, size=n)

# %%
ones = np.sum(sample == 1)
zeros = np.sum(sample == 0)
p_mle = ones / n
p_mle

# %%
h = 1e-4

# %%
ll_center = log_likelihood(p_mle,sample)
ll_right = log_likelihood(p_mle + h,sample)
ll_left = log_likelihood(p_mle - h,sample)

# %%
numerical_second_derivative = (ll_right - 2 * ll_center + ll_left) / (h ** 2)
computer_observed_info = -numerical_second_derivative

# %%
print(f"MLE p: {p_mle:.4f}")
print(f"Computer-calculated Curvature (2nd deriv): {numerical_second_derivative:.2f}")
print(f"Computer-calculated Observed Info:        {computer_observed_info:.2f}")
print(f"Theoretical Fisher Information:           {n / (TRUE_P * (1 - TRUE_P)):.2f}")

# %% [markdown]
r"""
## Log-odds
"""

# %%
p_candidates = np.linspace(0.05, 0.95, 200)
fisher_curvatures = 1 / (p_candidates * (1 - p_candidates))

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(p_candidates, fisher_curvatures)
ax.set_xlabel("Probabilities")
ax.set_ylabel("Fisher Information")
plt.show()

# %%
p_exact = np.linspace(0.5, TRUE_P, 200)
exact_curvatures = 1 / (p_exact * (1 - p_exact))

area_exact = simpson(exact_curvatures, x=p_exact)
formulatic_log_odds = np.log(TRUE_P / (1 - TRUE_P))

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(p_candidates, fisher_curvatures, color="navy")
ax.fill_between(
    p_exact,
    exact_curvatures,
    color="crimson",
    alpha=0.4,
    label=f"Exact Area = {area_exact:.4f}"
)

ax.axvline(0.5, color="gray", linestyle="--", label="0.5 Baseline")
ax.axvline(TRUE_P, color="crimson", linestyle=":", label=f"TRUE_P = {TRUE_P}")
ax.set_xlabel("Probabilities")
ax.set_ylabel("Fisher Information")
ax.legend()
plt.show()

# %%
print("Area under curve:", area_exact)
print("Formulatic log-odds:", formulatic_log_odds)
