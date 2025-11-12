import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_points_on_line(df, column_name):
    data = df[column_name].values
    y = [0] * len(data)

    plt.figure(figsize=(10, 2))
    sns.scatterplot(x=data, y=y, s=100)
    plt.yticks([])
    plt.xlabel(column_name)
    plt.tight_layout()
    plt.show()

def plot_points_2d(df, x_column, y_column):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x_column, y=y_column, s=100)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()

def plot_kmeans_step(points, labels, centroids, iteration=0, column_names=None):
    """
    Visualize k-means step.
    - points: array-like shape (n_samples, n_features) or (n_samples,) for 1D
    - labels: array-like of ints length n_samples
    - centroids: array-like shape (k, n_features) or (k,) for 1D
    - iteration: integer used for title
    - column_names: list like ['X','Y'] or ['Feature'] for axis labels
    """
    pts = np.asarray(points)
    ctr = np.asarray(centroids)
    lbl = np.asarray(labels, dtype=int)
    n_clusters = int(lbl.max()) + 1 if lbl.size else 0
    palette = sns.color_palette("husl", max(1, n_clusters))

    if pts.ndim == 1 or (pts.ndim == 2 and pts.shape[1] == 1):
        # 1D plot on a horizontal line
        x = pts.flatten()
        y = np.zeros_like(x, dtype=float)
        plt.figure(figsize=(10, 2.5))
        # color each point by its label
        colors = [palette[l] for l in lbl]
        plt.scatter(x, y, c=colors, s=120, edgecolor='k', zorder=2)
        # annotate with index
        for i, xi in enumerate(x):
            plt.annotate(str(i), (xi, 0), xytext=(0, 12), textcoords='offset points',
                         ha='center', fontsize=9, weight='bold')
        # centroids
        ctr_vals = ctr.flatten()
        plt.scatter(ctr_vals, np.zeros_like(ctr_vals), c='red', marker='X', s=220,
                    edgecolor='black', linewidth=1.5, zorder=3, label='centroids')
        plt.yticks([])
        plt.xlabel(column_names[0] if column_names else 'Value')
        plt.title(f'K-Means Iteration {iteration}')
        plt.tight_layout()
        plt.show()
        return

    # 2D visualization
    if pts.ndim >= 2 and pts.shape[1] >= 2:
        x = pts[:, 0]
        y = pts[:, 1]
        plt.figure(figsize=(8, 6))
        colors = [palette[l] for l in lbl]
        plt.scatter(x, y, c=colors, s=120, edgecolor='k', zorder=2)
        # annotate with index
        for i in range(len(pts)):
            plt.annotate(str(i), (x[i], y[i]), xytext=(6, 6), textcoords='offset points',
                         fontsize=9, weight='bold')
        # plot centroids
        if ctr.ndim == 1:
            plt.scatter(ctr[0], ctr[1], c='red', marker='X', s=260, edgecolor='black', linewidth=1.5, zorder=3)
        else:
            plt.scatter(ctr[:, 0], ctr[:, 1], c='red', marker='X', s=260,
                        edgecolor='black', linewidth=1.5, zorder=3, label='centroids')
        if column_names and len(column_names) >= 2:
            plt.xlabel(column_names[0]); plt.ylabel(column_names[1])
        else:
            plt.xlabel('X'); plt.ylabel('Y')
        plt.title(f'K-Means Iteration {iteration}')
        plt.tight_layout()
        plt.show()
        return

    # fallback: plot as scatter
    plt.figure(figsize=(8, 6))
    colors = [palette[l] for l in lbl]
    plt.scatter(pts[:, 0], pts[:, 1], c=colors, s=120, edgecolor='k')
    plt.title(f'K-Means Iteration {iteration}')
    plt.tight_layout()
    plt.show()
