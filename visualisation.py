import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from toolbox import load_otto_db


def visualisation(X, y, manifold='pca', n_components=2,class):
    """
    @brief: visualise features of data manifolded in 2 or 3 dimensions, with PCA or TSNE

    @param:
            X: ndarray, (n_samples, n_features), the data to manifold
            y: ndarray, (n_samples,), the corresponding targets
            manifold: 'pca' or 'tsne', choice of the algorithm to compress the features
            n_components: dimension of the destination space, for the manifold (can be 2 or 3)
    """

    print('Training Manifold...')

    if manifold == 'tsne':
        X_manifold = TSNE(n_components=n_components, verbose=10).fit_transform(X)
    elif manifold == 'pca':
        X_manifold = PCA(n_components=n_components).fit_transform(X)

    print('End training')

    fig = plt.figure()

    if n_components not in (2, 3):
        raise ValueError('Too high dimension, no visualisation possible for now')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    elif n_components == 2:
        ax = fig.add_subplot(111)

    for class_i in np.unique(y):
        if n_components == 2:
            ax.scatter(X_manifold[y == class_i, 0],
                       X_manifold[y == class_i, 1],
                       label=str(class_i))
        elif n_components == 3:
            ax.scatter(X_manifold[y_train == class_i, 0],
                       X_manifold[y_train == class_i, 1],
                       X_manifold[y_train == class_i, 2],
                       label=str(class_i))

    plt.legend()
    plt.show()

    return X_manifold


if __name__ == "__main__":

    # load datasets
    X, y = load_otto_db()
    X_test = load_otto_db(test=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # run visualisation
    visualisation(X_train[:10000], y_train[:10000], manifold='tsne', n_components=2)
