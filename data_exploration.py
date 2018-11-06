import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from toolbox import load_otto_db


def visualisation(X, y, manifold='pca', n_components=2, normalisation=True):
    """
    @brief: Visualize features of data manifolded in 2 or 3 dimensions, with PCA or TSNE

    @param:
            X: ndarray, (n_samples, n_features), the data to manifold
            y: ndarray, (n_samples,), the corresponding targets
            manifold: 'pca' or 'tsne', choice of the algorithm to compress the features
            n_components: dimension of the destination space, for the manifold (can be 2 or 3)
    """

    print('Training Manifold...')

    transformers = []

    if normalisation:
        transformers = [('scaling', StandardScaler())]

    if manifold == 'tsne':
        transformers.append(('tsne', TSNE(n_components=n_components, verbose=10)))
        pipeline = Pipeline(transformers)
    elif manifold == 'pca':
        transformers.append(('pca', PCA(n_components=n_components)))
        pipeline = Pipeline(transformers)

    X_manifold = pipeline.fit_transform(X)

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
                       label=str(class_i),
                       alpha=0.25)
        elif n_components == 3:
            ax.scatter(X_manifold[y_train == class_i, 0],
                       X_manifold[y_train == class_i, 1],
                       X_manifold[y_train == class_i, 2],
                       label=str(class_i),
                       alpha=0.25)

    plt.legend()
    plt.title('{} in {} dimensions'.format(manifold, n_components))

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
    plt.xticks(range(X.shape[1]), indices, rotation='horizontal', size=8)
    plt.xlim([-1, X.shape[1]])

    indices_best_features = np.arange(len(importances))[importances > threshold]

    if return_model:
        return indices_best_features, forest
    else:
        return indices_best_features


if __name__ == '__main__':

    # load data
    print('-------------------------------------------------------')
    print("Loading dataset...")
    X, y = load_otto_db()

    # print('WARNING: reducing numbers of samples to relieve RAM')
    # pe   rm = np.arange(X.shape[0])
    # np.random.shuffle(perm)
    # X, y = X[perm, :], y[perm]
    # X, y = X[:1000, :], y[:1000] # You can remove thoses lines if you have a better computer

    # separate training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

    ###############################################
    ####### Correlation between features ##########
    ###############################################

    ## Look for correlation between the features to see if there is redondancy in the data or not

    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.matshow(corr_matrix, cmap='seismic')
    plt.colorbar()
    plt.title('Correlation matrix of features in the dataset')

    #####################################
    ####### Feature importance ##########
    #####################################

    # The idea of this section is to reduce the features used for learning, by selecting only
    ## those who seems to help in the classification (in a attempt to reduce noise)

    print('-------------------------------------------------------')
    print("Computing Features importance...")
    indices_best_features, forest = feature_importance(X_train,
                                                       y_train,
                                                       return_model=True)
    print('TEST SCORE, ALL FEATURES: ', forest.score(X_valid, y_valid))

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
    # You can choose you projection method (PCA or T-SNE)
    ## and the number of reduced dimension (2 or 3)
    visualisation(X_train, y_train, manifold='tsne', n_components=2)

    plt.show()
