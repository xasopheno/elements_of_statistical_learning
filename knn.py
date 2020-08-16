print("knn")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from data import make_data
from plot import plot


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=5, weights="uniform", n_classes=2):
        self.X_train = X_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.n_classes = n_classes

    def euclidean_distance(self, a: np.array, b: np.array):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def kneighbors(self, X_test, return_distance=False):
        dist = []
        neigh_ind = []

        point_dist = [
            self.euclidean_distance(x_test, self.X_train) for x_test in X_test
        ]

        print("\npoint_dist: \n", len(point_dist[0]), len(point_dist))

        for row in point_dist:
            enum_neigh = enumerate(row)

            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[: self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]
            dist.append(dist_list)
            neigh_ind.append(ind_list)

        print("\ndist: \n", len(dist[0]), len(dist))
        print("\nneigh_ind: \n", len(dist[0]), len(dist))
        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X_test):
        if self.weights == "uniform":
            neighbors = self.kneighbors(X_test)

            y_hat = np.array(
                [
                    np.argmax(np.bincount(self.y_train[neighbor].astype(int)))
                    for neighbor in neighbors
                ]
            )

            print("\ny_hat\n", len(y_hat))
            return y_hat
        if self.weights == "distance":
            dist, neigh_ind = self.kneighbors(X_test, return_distance=True)

            inv_dist = 1 / dist
            mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]

            proba = []

            for i, row in enumerate(mean_inv_dist):
                row_pred = self.y_train[neigh_ind[i]]

                for k in range(self.n_classes):
                    indicies = np.where(row_pred == k)
                    prob_ind = np.sum(row[indicies])
                    proba.append(np.array(prob_ind))

            predict_proba = np.array(proba).reshape(X_test.shape[0], self.n_classes)

            y_hat = np.array([np.argmax(item) for item in predict_proba])

            return y_hat
        else:
            exit(
                NameError(
                    "ERROR: Use 'uniform' or 'distance' for weights in KNN initialization"
                )
            )

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        return float(sum(y_pred == y_test)) / float(len(y_test))


n_neighbors = 21
n_classes = 2

X_train, y_train, X_test, y_test = make_data(n_samples=1000)

knn = KNN(X_train, y_train, n_neighbors=n_neighbors, weights="distance")
plot(knn, X_test, y_test)
score = knn.score(X_test, y_test)
print(score)
