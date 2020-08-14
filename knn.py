print("knn.py")

import numpy as np

import pandas as pd
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame


X, y = make_blobs(
    n_samples=10, centers=2, n_features=2, random_state=1, cluster_std=7.0
)

data = np.hstack((X, y[:, np.newaxis]))

#  np.random.shuffle(data)

split_rate = 0.7

train, test = np.split(data, [int(split_rate * (data.shape[0]))])

X_train = train[:, :-1]
y_train = train[:, -1]

X_test = test[:, :-1]
y_test = test[:, -1]

print(X_train)
print(X_test)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: "orange", 1: "green"}
fig, ax = pyplot.subplots()
grouped = df.groupby("label")
for key, group in grouped:
    group.plot(ax=ax, kind="scatter", x="x", y="y", label=key, color=colors[key])
#  pyplot.show()


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=5, weights="uniform"):
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_classes = 3

    def euclidean_distance(self, a: np.array, b: np.array):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def knn(self, X_test, return_distance=False):
        dist = []
        neigh_ind = []

        point_dist = [
            self.euclidean_distance(x_test, self.X_train) for x_test in X_test
        ]


#  a = np.array([1, 1, 1, 0])
#  b = np.array([[1, 1, 1, 1], [1, 1, 1, 0]])

#  print(euclidean_distance(a, b))
#  assert np.all(euclidean_distance(a, b) == np.array([1.0, 0.0]))
