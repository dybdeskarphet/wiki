# %% [markdown]
r"""
---
title: "What is Fisher Information?"
date: 2026-05-28
tags: [statistics, fisher-information, likelihood, variance]
description: "Mathematical definition and intuition behind Fisher Information and the Cramér-Rao lower bound."
---
"""

# %% [markdown]
r"""
# What is Fisher information?

## Maximum Likelihood Estimation
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %%
TRUE_MU = 85.0
KNOWN_SIGMA = 12.0

# %% [markdown]
r"""
First, we determine the population mean and standard deviation.
"""

# %%
api_logs = np.random.normal(loc=TRUE_MU, scale=KNOWN_SIGMA, size=5)
api_logs_50 = np.random.normal(loc=TRUE_MU, scale=KNOWN_SIGMA, size=50)
print(f"Recorded API response times (ms) (n=5): {np.round(api_logs, 2)}")
print(f"Recorded API response times (ms) (n=50): {np.round(api_logs, 2)}")

# %% [markdown]
r"""
We draw two samples from our population with sample sizes of 5 and 50. In these lines, we have also indicated that our population is normally distributed.
"""

# %%
mle_mu = np.mean(api_logs)
mle_mu_50 = np.mean(api_logs_50)
print(f"Sample mean (MLE of mu) (n=5) {mle_mu}")
print(f"Sample mean (MLE of mu) (n=50): {mle_mu_50}")

# %% [markdown]
r"""
As we learned in the section on maximum likelihood estimators, we know that the sample mean is the maximum likelihood estimator for the mean of population, therefore, we define the maximum likelihood estimators for the population mean of both samples as the sample means.
"""

# %%
x_axis = np.linspace(40, 130, 500)

# %%
pdf_curve = norm.pdf(x_axis, loc=mle_mu, scale=KNOWN_SIGMA)
pdf_curve_50 = norm.pdf(x_axis, loc=mle_mu_50, scale=KNOWN_SIGMA)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].set_title("n=5")
axes[0].plot(x_axis, pdf_curve)
axes[0].axvline(mle_mu, label="Sample mean", color="green", alpha=0.3)
axes[0].axvline(TRUE_MU, label="Population mean", color="red", alpha=0.3)
axes[0].scatter(api_logs, np.zeros_like(api_logs), color="green", alpha=0.3)
axes[0].legend()
axes[1].set_title("n=50")
axes[1].plot(x_axis, pdf_curve_50)
axes[1].axvline(mle_mu_50, label="Sample mean", color="green", alpha=0.3)
axes[1].axvline(TRUE_MU, label="Population mean", color="red", alpha=0.3)
axes[1].scatter(api_logs_50, np.zeros_like(api_logs_50), color="green", alpha=0.3)
axes[1].legend()
plt.show()

# %% [markdown]
r"""
What we’ll learn about Fisher information will also give us some insight into sample size. But as we can see here, an increase in sample size does indeed indicate that we’re getting closer to the population mean.
"""

# %% [markdown]
r"""
Now let’s generate 500 linearly spaced guesses for the population mean, ranging from 0 to 150.
"""

# %%
mu_guesses = np.linspace(0, 150, 500)

# %%
api_logs

# %% [markdown]
r"""
"""

# %% [markdown]
r"""
The mathematical equivalents of the relevant lines are as follows (for the likelihood function)

$$
\begin{aligned}
\mathtt{\text{norm.pdf(api_logs, ...)}} : f(x_i \mid \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right) \\\\
\mathtt{np.prod(...)} : L(\mu) = \prod_{i=1}^{5} f(x_i \mid \mu, \sigma) \\\\
\mathtt{\text{raw_likelihoods}} = \begin{bmatrix} L(\mu_1) \\\\ L(\mu_2) \\\\ \vdots \\\\ L(\mu_{500}) \end{bmatrix}
\end{aligned}
$$
"""

# %%
raw_likelihoods = np.array(
    [np.prod(norm.pdf(api_logs, loc=guess, scale=KNOWN_SIGMA)) for guess in mu_guesses]
)
raw_likelihoods[:5]

# %% [markdown]
r"""
And for the log-likelihood function, we have the following.

$$
\begin{aligned}
\mathtt{np.sum(...)} : l(\mu) = \sum_{i=1}^{5} \ln\big(f(x_i \mid \mu, \sigma)\big) \\\\
\mathtt{\text{log_likelihoods}} = \begin{bmatrix} \ell(\mu_1) \\\\ \ell(\mu_2) \\\\ \vdots \\\\ \ell(\mu_{500}) \end{bmatrix}
\end{aligned}
$$
"""

# %%
log_likelihoods = np.array(
    [
        np.sum(norm.logpdf(api_logs, loc=guess, scale=KNOWN_SIGMA))
        for guess in mu_guesses
    ]
)
log_likelihoods[:5]

# %%
log_likelihoods_50 = np.array(
    [
        np.sum(norm.logpdf(api_logs_50, loc=guess, scale=KNOWN_SIGMA))
        for guess in mu_guesses
    ]
)
log_likelihoods_50[:5]

# %% [markdown]
r"""
Since we are primarily interested in the log-likelihood function, we calculated the results of the log-likelihood function for all our $\mu$ guesses for our 50 variable sample as well.
"""

# %% [markdown]
r"""
For our sample consisting of 5 items, the likelihood and log-likelihood functions produce the following plots for all mu guesses.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].set_title("n=5")
axes[0].plot(mu_guesses, raw_likelihoods)
axes[0].axvline(mle_mu, color="black", linestyle="--", label=f"Peak = {mle_mu:.1f}ms")
axes[0].set_xlabel("Guessed average latency (ms)")
axes[0].set_ylabel("Probability Product $L(\\theta)=\prod_{i=1}^{n} f(x_i| \\theta)$")
axes[0].legend()

axes[1].set_title("n=5")
axes[1].plot(mu_guesses, log_likelihoods, color="navy")
axes[1].axvline(mle_mu, color="black", linestyle="--", label=f"Peak = {mle_mu:.1f}ms")
axes[1].set_xlabel("Guessed average latency (ms)")
axes[1].set_ylabel("Sum of Log-Probabilities $\sum_{i=1}^{n}ln(f(x_i|\\theta))$")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
Since the values on the y-axis here do not represent a probability density function, they are not particularly meaningful. What we should focus on is where the peaks of these functions lie.

The peak points of the graphs of the two functions also correspond to the sample mean, which is something we already know from MLE notes.
"""

# %%
fig, ax = plt.subplots()

plt.plot(mu_guesses, log_likelihoods, color="black", label="initial $\\ell$ (n=5)")
plt.plot(
    mu_guesses,
    log_likelihoods_50,
    color="darkorange",
    linestyle="-",
    label=f"$\ell$ when n=50",
)
plt.legend()
plt.show()

# %% [markdown]
r"""
Here we also see that while the peak of the 5-element sample is not very sharp, the peak of the 50-element sample is quite sharp. It makes sense to place more confidence in the estimate with the sharper peak. For our sample of 50 elements, this sharp peak formed by the $\mu$ guesses entered into our log-likelihood function indicates high Fisher information, but how?
"""

# %% [markdown]
r"""
## Fisher information
"""

# %% [markdown]
r"""
In the previous steps, we created samples containing 5 and 50 elements only once. Here, however, we created a large number of samples with 50 elements (in practice, an infinite number of times, provided the number of digits after the decimal point is not too many).
"""

# %%
sample_size_50_universes = np.random.normal(
    loc=TRUE_MU, scale=KNOWN_SIGMA, size=(5000, 50)
)

# %% [markdown]
r"""
And we know that the sample mean is the maximum likelihood estimator of the population mean, so we collected the means of all the samples we have into a new array.
"""

# %%
means_of_universes = np.mean(sample_size_50_universes, axis=1)

# %% [markdown]
r"""
Now let’s calculate the variance of these sample means we’ve obtained or in other words, let’s calculate the variance of the sampling distribution of the sample mean.
"""

# %%
brute_force_variance = np.var(means_of_universes)

# %% [markdown]
r"""
If the sample means we obtain from the samples differ significantly from one another (variance), this leads us to conclude that our sample size is insufficient to make reliable inferences, and that the sample mean, while an unbiased estimator, is not a very reliable estimator.
"""

# %% [markdown]
r"""
So a high variance indicates that we have little information, while a low variance indicates that we have a lot of information. Now, keep this in mind.
"""

# %%
fisher_information = 50 / KNOWN_SIGMA**2

# %% [markdown]
r"""
The Fisher information of the entire sample with known variance is $\frac{n}{\sigma^2}$.
"""

# %%
print(f"Inverse variance of the estimator (sample mean): {1/brute_force_variance}")
print(f"Thoretical Fisher information: {fisher_information}")

# %% [markdown]
r"""
We know that the sampling distribution of the sample mean has a variance of $\frac{\sigma^2}{n}$. The inverse of the variance of our estimator is nearly equal to the Fisher information of this distribution, which essentially indicates how much information we have. This is because Cramér-Rao Lower Bound states that variance of any unbiased estimator must be greater or equal to the inverse of the Fisher information.
"""
