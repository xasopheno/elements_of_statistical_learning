import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot(model, X_test: np.array, y_test: np.array, grid_step=0.2):
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"])
    # calculate min, max and limits
    x_min, x_max = X_test.min() - 1, X_test.max() + 1
    y_min, y_max = y_test.min() - 1, y_test.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
    )

    # predict class using data and kNN classifier
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Classification")
    plt.show()

