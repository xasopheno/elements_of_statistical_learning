print("least_squares")
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from data import make_data
from plot import plot

#  (X^-1 * X.T) * y
# X * X.T
# [
#   [1, 3],
#   [2, 4]
#  ... ]

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

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, 3])

#  y.T * y = norm^2


print(y.T)
print(m.T)
print(m)
print(y)

#  v.T * v =inner product

#  v.T * m.T * m * v

#  x = m * y
#  x.T = y.T * m.T
#  x.T * x = norm of y in m.T space

#  print(y.T * (m.T * m) * y)

#  y.T * y = norm_squared of y
#  m.T * norm_y * m = norm of y in vector space of m

#  print(a)
#  print(a.T)
#  print(a.T.dot(a))
#  print(inv(a.T.dot(a)))
#  print(inv(a).dot(a))


#  knn = KNN(X_train, y_train, n_neighbors=n_neighbors, weights="distance")
#  plot(knn, X_test, y_test)
#  score = knn.score(X_test, y_test)
#  print(score)
