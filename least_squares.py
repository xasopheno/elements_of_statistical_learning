print("least_squares")
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from data import make_data
from plot import plot


class LeastSquares:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.beta_hat = None

    def find_coeffs(self):
        #  β=(X.T*X)^−1 * X.T * y
        intercept = np.ones(shape=self.y_train.shape).reshape(-1, 1)
        self.X_train = np.concatenate((X_train, intercept), 1)

        self.beta_hat = (
            inv(self.X_train.transpose().dot(self.X_train))
            .dot(self.X_train.transpose())
            .dot(self.y_train)
        )

        print("\nβ_hat\n", self.beta_hat)
        intercept = np.ones(shape=self.y_train.shape).reshape(-1, 1)

    def predict(self, X_test):
        intercept = np.ones(shape=X_test.shape[0]).reshape(-1, 1)
        X_test = np.concatenate((X_test, intercept), 1)

        return X_test.dot(self.beta_hat)


X_train, y_train, X_test, y_test = make_data(n_samples=10000)

least_squares = LeastSquares(X_train, y_train)
least_squares.find_coeffs()
plot(least_squares, X_test, y_test)
#  score = knn.score(X_test, y_test)
#  print(score)

