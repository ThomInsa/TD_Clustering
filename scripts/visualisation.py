import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def plot_feature_distributions(df=None, csv_path=None, features=None, kind='kde',
                               bins=30, per_row=3, overlay=False, figsize=None,
                               palette='husl', log_scale=False, legend=True,
                               bw_method=None):
    """
    Plot distribution for each numeric feature in the Country-data.csv (or provided DataFrame).

    Parameters:
    - df: pandas.DataFrame or None. If None, csv_path must be provided.
    - csv_path: path to CSV file to load if df is None.
    - features: list of column names to plot. Defaults to all numeric cols except 'country'.
    - kind: 'kde' or 'hist' (kde uses sns.kdeplot, hist uses sns.histplot).
    - bins: number of bins for histplots.
    - per_row: number of subplots per row when not overlaying.
    - overlay: if True, overlay all feature distributions on one Axes with a legend.
    - figsize: tuple for figure size; if None it's auto-calculated.
    - palette: seaborn palette name or list of colors.
    - log_scale: if True, set x-axis to log scale.
    - legend: whether to show legend.
    - bw_method: passed to sns.kdeplot as bw_method (None uses default).
    """
    if df is None:
        if csv_path is None:
            raise ValueError("Either 'df' or 'csv_path' must be provided.")
        df = pd.read_csv(csv_path)

    # choose numeric features, exclude 'country' (case-insensitive)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if features is None:
        features = [c for c in numeric_cols if c.lower() != 'country']
    else:
        # ensure provided features exist
        features = [f for f in features if f in df.columns]

    if not features:
        raise ValueError("No numeric features to plot.")

    n = len(features)
    colors = sns.color_palette(palette, n)

    if overlay:
        plt.figure(figsize=figsize or (10, 6))
        ax = plt.gca()
        for i, feat in enumerate(features):
            series = df[feat].dropna()
            label = f"{feat} (n={series.size})"
            if kind == 'kde':
                sns.kdeplot(series, ax=ax, label=label, color=colors[i],
                            bw_method=bw_method, fill=False)
            else:
                sns.histplot(series, ax=ax, label=label, color=colors[i],
                             bins=bins, stat='density', alpha=0.4)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density' if kind == 'kde' or kind == 'hist' else '')
        if log_scale:
            ax.set_xscale('log')
        if legend:
            ax.legend(title='Feature (count)', fontsize='small', frameon=True)
        plt.title('Feature distributions (overlay)')
        plt.tight_layout()
        plt.show()
        return

    # grid of individual plots
    cols = max(1, per_row)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (4 * cols, 3.2 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for idx, feat in enumerate(features):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        series = df[feat].dropna()
        label = f"{feat} (n={series.size}, mean={series.mean():.2f})"
        if kind == 'kde':
            sns.kdeplot(series, ax=ax, color=colors[idx], bw_method=bw_method, fill=True, alpha=0.6)
            ax.set_ylabel('Density')
        else:
            sns.histplot(series, ax=ax, color=colors[idx], bins=bins)
            ax.set_ylabel('Count')
        ax.set_xlabel(feat)
        ax.set_title(feat)
        if log_scale:
            ax.set_xscale('log')
        if legend:
            ax.legend([label], fontsize='x-small', frameon=False)
    # hide unused subplots
    for j in range(n, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')
    plt.tight_layout()
    plt.show()
