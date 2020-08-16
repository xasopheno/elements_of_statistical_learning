import numpy as np
from sklearn.datasets import make_blobs, make_classification


def make_data(n_samples, n_classes=2):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_classes,
        random_state=0,
        cluster_std=5.0,
    )

    data = np.hstack((X, y[:, np.newaxis]))
    #  np.random.shuffle(data)

    split_rate = 0.7

    train, test = np.split(data, [int(split_rate * (data.shape[0]))])

    X_train = train[:, :-1]
    y_train = train[:, -1]
    print("len(train):", len(y_train))

    X_test = test[:, :-1]
    y_test = test[:, -1]
    print("len(test):", len(y_test))

    return X_train, y_train, X_test, y_test
