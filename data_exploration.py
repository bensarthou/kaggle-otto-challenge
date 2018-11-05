import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from otto_challenge import load_db


def visualisation(X, y, manifold='pca', n_components=2):
    """
    @brief: Visualize features of data manifolded in 2 or 3 dimensions, with PCA or TSNE

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



def feature_importance(X, y, threshold=0.01, return_model=False):

    """
    @brief: Select the most important features (according to a threshold) by training a tree
     classifier on the data and choose the features most used for splitting

    @params:
        X: ndarray, (n_samples, n_features), data for training
        y: ndarray, (n_samples, ), labels for the samples

        threshold: float, threshold to select the most important feature (in percentage of splits)
        return_model: bool, return the model used for computing importances

    @return:
        indices_best_features: ndarray 1d, indices of features to select in a dataset
        (if return_model)
        model: sklearn tree model, used for computing the importances
    """

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices, rotation='horizontal')
    plt.xlim([-1, X.shape[1]])
    plt.show()

    indices_best_features = np.arange(len(importances))[importances > threshold]

    if return_model:
        return indices_best_features, forest
    else:
        return indices_best_features


if __name__ == '__main__':

    X, y = load_db('data/train.csv')

    print('WARNING: reducing numbers of samples to relieve RAM')
    X, y = X[:10000, :], y[:10000] # You can remove this line if you have a better computer

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


    #####################################
    ####### Feature importance ##########
    #####################################

    # The idea of this section is to reduce the features used for learning, by selecting only
    ## those who seems to help in the classification (in a attempt to reduce noise)

    indices_best_features, forest = feature_importance(X_train,
                                                       y_train,
                                                       return_model=True)
    print('----------------------------------')
    print('TEST SCORE, ALL FEATURES: ', forest.score(X_valid, y_valid))
    print('----------------------------------')

    # Compare the validation score with a learning only on the "important" features
    forest_feature = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest_feature.fit(X_train[:, indices_best_features], y_train)
    print('TEST SCORE, MAIN FEATURES: ', forest_feature.score(X_valid[:, indices_best_features],
                                                              y_valid))


    #####################################
    ####### Visualisation ###############
    #####################################

    # With a difficult problem, visualising the data can be useful to determine which solutions
    ## can be best, but the issue is to reduce the dimensions of the data to a human-readable size

    print('-------------------------------------------------------')
    print('VISUALISATION OF DATA BY PROJECTING IN SMALL DIMENSIONS')
    # You can choose you projection method (PCA or T-SNE) and the number of reduced dimension (2 or 3)
    visualisation(X_train, y_train, manifold='tsne', n_components=2)
