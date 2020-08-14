print("knn.py")
from sklearn.datasets import make_blobs, make_classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 21
n_classes = 2


X, y = make_blobs(
    n_samples=1000,
    centers=n_classes,
    n_features=n_classes,
    random_state=0,
    cluster_std=5.0,
)

data = np.hstack((X, y[:, np.newaxis]))
np.random.shuffle(data)

split_rate = 0.7

train, test = np.split(data, [int(split_rate * (data.shape[0]))])

X_train = train[:, :-1]
y_train = train[:, -1]
print("len(train):", len(y_train))

X_test = test[:, :-1]
y_test = test[:, -1]
print("len(test):", len(y_test))


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


def plot(model: KNN, X_test: np.array, y_test: np.array, grid_step=0.2):
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"])
    # calculate min, max and limits
    x_min, x_max = X_test.min() - 1, X_test.max() + 1
    y_min, y_max = y_test.min() - 1, y_test.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
    )

    # predict class using data and kNN classifier
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points

    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i)" % (n_neighbors))
    plt.show()


knn = KNN(X_train, y_train, n_neighbors=n_neighbors, weights="distance")
plot(knn, X_test, y_test)
score = knn.score(X_test, y_test)
print(score)
