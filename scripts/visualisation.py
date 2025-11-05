import seaborn as sns
import matplotlib.pyplot as plt


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
