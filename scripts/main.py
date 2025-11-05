import numpy as np
import pandas as pd

def k_means(points_collection, cluster_number, initial_centroids=None, max_iter=100, tol=1e-6):
    points = np.asarray(points_collection, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    n_samples, n_features = points.shape

    if initial_centroids is None:
        rng = np.random.default_rng()
        choices = rng.choice(n_samples, cluster_number, replace=False)
        centroids = points[choices].astype(float)
    else:
        centroids = np.asarray(initial_centroids, dtype=float)
        if centroids.ndim == 1:
            # if user provided a single centroid vector or a flat list
            if centroids.size == n_features:
                centroids = centroids.reshape(1, -1)
            else:
                centroids = centroids.reshape(-1, 1)
        if centroids.shape[0] != cluster_number:
            # try to reshape if possible
            if centroids.size == cluster_number * n_features:
                centroids = centroids.reshape(cluster_number, n_features)
            else:
                raise ValueError("initial_centroids must match (cluster_number, n_features)")

    for _ in range(max_iter):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.empty((cluster_number, n_features), dtype=float)
        for i in range(cluster_number):
            assigned = points[labels == i]
            if assigned.size:
                new_centroids[i] = assigned.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(new_centroids, centroids, atol=tol):
            centroids = new_centroids
            break
        centroids = new_centroids

    # ensure labels correspond to final centroids
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)

    clusters = [points[labels == i] for i in range(cluster_number)]
    sse = sum_of_squared_errors(clusters)

    return centroids, sse


def sum_of_squared_errors(clusters):
    sse = 0.0
    for cluster in clusters:
        cluster = np.asarray(cluster, dtype=float)
        if cluster.size == 0:
            continue
        centroid = cluster.mean(axis=0)
        diffs = cluster - centroid
        sse += np.sum(np.sum(diffs * diffs, axis=1))
    return sse
