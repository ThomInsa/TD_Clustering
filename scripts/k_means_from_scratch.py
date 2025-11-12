import numpy as np
from scripts.visualisation import plot_kmeans_step

def k_means(points_collection, cluster_number, initial_centroids=None, max_iter=100, tol=1e-6,
            visualize=False, column_names=None):
    points = np.asarray(points_collection, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    n_samples, n_features = points.shape

    centroids = _initialize_centroids(points, cluster_number, initial_centroids)
    print("Initialized centroids:")
    print(centroids)

    if visualize:
        labels, _ = _get_labels_and_distances(points, centroids)
        plot_kmeans_step(points, labels, centroids, iteration=0, column_names=column_names)

    for it in range(max_iter):
        print(f"Iteration {it + 1} / {max_iter}")
        labels, _ = _get_labels_and_distances(points, centroids)

        counts = np.bincount(labels, minlength=cluster_number)
        print(" Cluster sizes:", counts)

        new_centroids = _update_centroids(points, labels, cluster_number, n_features, centroids)

        shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        max_shift = shifts.max() if shifts.size else 0.0
        print(" Centroid shifts:", shifts, "max:", max_shift)

        if visualize:
            plot_kmeans_step(points, labels, new_centroids, iteration=it + 1,
                             column_names=column_names)

        if np.allclose(new_centroids, centroids, atol=tol):
            centroids = new_centroids
            print(f"Converged at iteration {it + 1}, max shift {max_shift:.6g}")
            break
        centroids = new_centroids

    labels, _ = _get_labels_and_distances(points, centroids)
    clusters = [points[labels == i] for i in range(cluster_number)]
    sse = sum_of_squared_errors(clusters)

    print("Final centroids:")
    print(centroids)
    print(f"Final SSE: {sse:.6g}")

    return centroids, sse


def _initialize_centroids(points, cluster_number, initial_centroids):
    n_samples, n_features = points.shape

    if initial_centroids is None:
        rng = np.random.default_rng()
        choices = rng.choice(n_samples, cluster_number, replace=False)
        return points[choices].astype(float)

    centroids = np.asarray(initial_centroids, dtype=float)
    if centroids.ndim == 1:
        if centroids.size == n_features:
            centroids = centroids.reshape(1, -1)
        else:
            centroids = centroids.reshape(-1, 1)
    if centroids.shape[0] != cluster_number:
        if centroids.size == cluster_number * n_features:
            centroids = centroids.reshape(cluster_number, n_features)
        else:
            raise ValueError("initial_centroids must match (cluster_number, n_features)")
    return centroids


def _get_labels_and_distances(points, centroids):
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1), distances


def _update_centroids(points, labels, cluster_number, n_features, old_centroids):
    new_centroids = np.empty((cluster_number, n_features), dtype=float)
    for i in range(cluster_number):
        assigned = points[labels == i]
        if assigned.size:
            new_centroids[i] = assigned.mean(axis=0)
        else:
            print(f"  Cluster {i} empty; keeping previous centroid")
            new_centroids[i] = old_centroids[i]
    return new_centroids

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
