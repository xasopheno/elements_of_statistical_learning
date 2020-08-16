print("least_squares")
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from data import make_data
from plot import plot

#  β=(X.T*X)^−1 * X.T * y

X_train, y_train, X_test, y_test = make_data(n_samples=6)

intercept = np.ones(shape=y_train.shape).reshape(-1, 1)
print("intercept:", intercept)
X_train = np.concatenate((X_train, intercept), 1)
print("\nX_train\n", X_train)
print(X_train.shape)

a = X_train.T
print("\nX_train.T\n", a)
print(a.shape)
a = a.dot(X_train)
print("\nX_train.T.dot(X_train)\n", a)
print(a.shape)
a = inv(a)
print("\nX_train.T.dot(X_train)*-1\n", a)
print(a.shape)

a = a.dot(X_train.T)
print("\nX_train.T * (X_train)*-1 * X_train.T\n", a)
a = a.dot(y_train)
print("\nX_train.T * (X_train)*-1 * X_train.T * y_train\n", a)

beta_hat = inv(X_train.transpose().dot(X_train)).dot(X_train.transpose()).dot(y_train)
print("\nbeta hat\n", beta_hat)

#  print("\nlinalg\n", np.linalg.lstsq(X_train, y_train, rcond=None))

print("\ny_hat = X_train * beta_hat)\n", X_train.dot(beta_hat))
print("\ny_train", y_train)
print("++++++++++++")

a = np.array([[1, 2], [4, 4]])
print(a)
print(a.T)
print(a.T.dot(a))
print(inv(a.T.dot(a)))
print(inv(a).dot(a))


#  knn = KNN(X_train, y_train, n_neighbors=n_neighbors, weights="distance")
#  plot(knn, X_test, y_test)
#  score = knn.score(X_test, y_test)
#  print(score)
