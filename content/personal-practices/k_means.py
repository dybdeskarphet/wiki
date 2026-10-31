# %% [markdown]
r"""
---
title: "K-means"
date: 2026-10-31
tags: [statistics, k-means, ml]
description: "Understanding how K-means and K-means++ works"
---
"""

# %% [markdown]
r"""
First, we import our libraries and set the random seed
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
np.random.seed(9)

# %% [markdown]
r"""
For this topic, we will use two-dimensional data to make it easier to visualize; that is, each data point will contain x and y coordinates.
"""

# %%
points_per_cluster = 200
spread = 2.0

# %% [markdown]
r"""
We have 3 centroids, and at each of these 3 centroids, we will generate 200 data points, for a total of 600 data points, with normal noise with mean of 0 and a standard deviation of 2.
"""

# %%
centers = pd.DataFrame(
    np.array([[2.0, 2.0], [8.0, 3.0], [5.0, 9.0]]), columns=["x", "y"]
)
real_centers = centers.copy()

# %%
clusters = [
    (c + np.random.randn(points_per_cluster, 2) * spread)
    for c in real_centers.to_numpy()
]

# %%
data = pd.DataFrame(np.vstack(clusters), columns=["x", "y"])
data["center_index"] = -1

# %%
print(data[:5])
print(data[200:205])
print(data[400:405])

# %% [markdown]
r"""
Here's how our data looks like on a scatter plot.
"""

# %%
fig, ax = plt.subplots()
ax.scatter(data["x"], data["y"], s=5)
plt.show()

# %% [markdown]
r"""
We are writing a function to calculate the distance. It will accept a point and a centroid as parameters and perform calculations using their x and y coordinates.
"""


# %%
def calculate_distance(point, center):
    return math.dist((point["x"], point["y"]), (center["x"], center["y"]))


# %%
old_assignments = []

# %% [markdown]
r"""
As you may recall, we initially identified three centroids. And we had a total of 600 points.

The distance from each of these points to each centroid is, of course, different, and each point is naturally closest to exactly one centroid.

If we define the centroid closest to each point as that point’s cluster, we have performed the first key step of the K-means algorithm.

As you remember, each of the three centroids also had coordinates that we initially determined.

We need to recalculate the coordinates of these three centroids so that we have the coordinates of the centroid of our current cluster. To do this:

For each cluster: The average of the x-coordinates of all points within it becomes the new x-coordinate of that centroid; the average of the y-coordinates becomes the new y-coordinate of that centroid.

Once we’ve done this, we’ve completed our first iteration. The next step will be to recalculate the nearest points based on the new centroid coordinates. And this iteration will continue until the centroid coordinates remain virtually unchanged.
"""

# %%
for iteration in range(50):
    assignments = []

    for _, p in data.iterrows():
        closest_ix = None
        closest_distance = float("inf")

        for ix_c, ct in centers.iterrows():
            distance = calculate_distance(p, ct)
            if distance < closest_distance:
                closest_distance = distance
                closest_ix = ix_c

        assignments.append(closest_ix)

    data["center_index"] = assignments

    for ix_c in range(len(centers)):
        group = data[data["center_index"] == ix_c]

        if len(group) > 0:
            centers.loc[ix_c, "x"] = group["x"].mean()
            centers.loc[ix_c, "y"] = group["y"].mean()

    if old_assignments is not None and np.array_equal(assignments, old_assignments):
        print(f"Converged early at iteration {iteration}!")
        break

    old_assignments = assignments

# %% [markdown]
r"""
As a result, we will end up with points grouped into specific clusters. Below, you can see each cluster in a different color, and each centroids marked with a black X.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(data["x"], data["y"], s=7)
sns.scatterplot(
    data=data, x="x", y="y", hue="center_index", palette="Set2", s=12, ax=axes[1]
)
sns.scatterplot(
    data=centers,
    x="x",
    y="y",
    marker="X",
    color="black",
    s=50,
    ax=axes[1],
    legend=True,
    label="centroids",
)
plt.show()

# %% [markdown]
r"""
## What if we didn't cheat on selecting the centroids?

In the above example, to make it easier to understand, we specifically chose clusters that are the same as the ones we initially used to generate the data. However, under normal circumstances, we would have no information about where the correct centers should be.

There are several ways to determine the centers, but for this example, we’ll use the K-means++ algorithm.
"""

# %%
k = 3

# %% [markdown]
r"""
- First of all, we are creating a new centroid store called `centers_plus`, we will use this to compare K-means++ centroids and our actual centers.
- After that, we select a random point from our dataset which will be our first center. For the first center, we are not using a smart way to select it, it will be our reference point for other centers.
- Now, we calculate the distance of each point to the centers we have (right now, we only have one), square them, and store them in a list so that we know which points are farther to any of the centroids and which points are closer.
- Important thing here is that, we don't care about index of the centroid it is close or far away from, we only care about the distance of it:
    - If `point_A` is 20 units away from `centroid_X`, but 22 units away from `centroid_Y`, we store the distance from `centroid_X`.
    - If `point_B` is 50 units away from `centroid_X`, but 10 units away from `centroid_Y`, we store the distance from `centroid_Y`, because we only care if the point is too close to any of the centroids.
- We assign the squared distances to each point as their weights, if they are far away from a centroid, they have much more weight.
- Finally, we select the next centroid from the data points we have, using squared_distances as their probability_weight for each.
    - This creates a situation where a point 200 units away from a centroid is most likely to get selected compared to a point 50 units away from a centroid, but still have the chance get selected.
    - The reason we don't select the farthest point is that, if there are outliers in the dataset, centroids always become those outliers which makes our algorithm fall apart.
- After selecting the centroid, we start again from calculating the distances, squaring them, and iterate the exact same steps until we find `k` centroids.
"""

# %%
data["center_index"] = -1
centers_plus = [data.sample(1).iloc[0]]
while len(centers_plus) < k:
    squared_distances = []
    for _, p in data.iterrows():
        min_dist = min(calculate_distance(p, c) for c in centers_plus)
        squared_distances.append(min_dist**2)

    next_center = data.sample(1, weights=squared_distances).iloc[0]
    centers_plus.append(next_center)

centers_plus = pd.DataFrame(centers_plus)[["x", "y"]].reset_index(drop=True)
centers_plus, real_centers


# %%

# %% [markdown]
r"""
So, if we hadn’t used Euclidean distance when calculating distance, or the mean when recalculating the center, what other methods would we have encountered? To find the answer, I recommend taking a closer look at the following topics:

- Spherical K-Means
- K-Medians
- K-Medoids
- Gaussian Mixture Models (GMM)
"""
