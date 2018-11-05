import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE

from toolbox import load_otto_db


if __name__ == "__main__":

    X, y = load_otto_db()
    X_test = load_otto_db(test=True)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training TSNE...')
    X_embedded = TSNE(n_components=2, verbose=10).fit_transform(X[:10])
    print('End training')
    plt.figure()
    l_colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
    for idx, class_i in enumerate(np.unique(y)):
        plt.scatter(X_embedded[y[:10000] == class_i, 0], X_embedded[y[:10000] == class_i, 1],
        label=str(y), color=l_colors[i])
    plt.legend()
    plt.show()
    # models = []
    #
    # ### SMALL EXS 1
    # # # Logistic Regression
    # # parameters = {'penalty':['l1'], 'C':[0.1, 1.]}
    # # models.append(("LogisticRegression", LogisticRegression(), parameters))
    # #
    # # # DecisionTree
    # # parameters = {'max_depth':[3, 10], 'min_samples_split':[2, 5]}
    # # models.append(("DecisionTree", DecisionTreeClassifier(), parameters))
    #
    # ## ALL MODEL, 1 PARAM
    #
    # # # SVM
    # # parameters = {'kernel':['poly'], 'C':[0.1], 'probability':[True]}
    # # models.append(("SVC", SVC(), parameters))
    #
    # # KNeighbors
    # parameters = {'n_neighbors':[3], 'leaf_size':[10], 'p':[1]}
    # models.append(("KNeighbors", KNeighborsClassifier(), parameters))
    #
    # # DecisionTree
    # parameters = {'splitter':['best', 'random'], 'criterion':['gini', 'entropy'],
    #               'max_depth':[3], 'min_samples_split':[2]}
    # models.append(("DecisionTree", DecisionTreeClassifier(), parameters))
    #
    # # RandomForestClassifier
    # parameters = {'n_estimators':[10], 'criterion':['gini'],
    #               'max_depth':[None], 'min_samples_split':[2]}
    # models.append(("RandomForest", RandomForestClassifier(), parameters))
    #
    # # GradientBoostingClassifier
    # parameters = {'loss':['exponential'], 'n_estimators':[10],
    #               'criterion':['mae'], 'max_depth':[3],
    #               'min_samples_split':[2]}
    # models.append(("GradientBoostingClassifier", GradientBoostingClassifier(), parameters))
    #
    # # Multi-Layer Classifier
    # parameters = {'hidden_layer_sizes':[(70,)],
    #               'activation':['logistic'],
    #               'batch_size':[128],
    #               'alpha':[0.001]}
    # models.append(("MLPClassifier", MLPClassifier(), parameters))
    #
    ### ALL PARAMS
    # Logistic Regression
    parameters = {'penalty':['l1', 'l2'], 'C':[0.1, 1., 10.]}
    models.append(("LogisticRegression", LogisticRegression(), parameters))

    # SVM
    parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'], 'C':[0.1, 1., 10., 100], 'probability':True}
    models.append(("SVC", SVC(), parameters))

    # KNeighbors
    parameters = {'n_neighbors':[3, 5, 7], 'leaf_size':[10, 20, 30, 40], 'p':[1, 2]}
    models.append(("KNeighbors", KNeighborsClassifier(), parameters))

    # DecisionTree
    parameters = {'splitter':['best', 'random'], 'criterion':['gini', 'entropy'],
                  'max_depth':[3, 5, 10, 20], 'min_samples_split':[2, 3, 5]}
    models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

    # RandomForestClassifier
    parameters = {'n_estimators':[10, 50, 100], 'criterion':['gini', 'entropy'],
                  'max_depth':[None, 3, 10], 'min_samples_split':[2, 3, 5]}
    models.append(("RandomForest", RandomForestClassifier(), parameters))

    # GradientBoostingClassifier
    parameters = {'loss':['deviance', 'exponential'], 'n_estimators':[10, 50, 100],
                  'criterion':['friedman_mse', 'mae'], 'max_depth':[3, 10],
                  'min_samples_split':[2, 3, 5]}
    models.append(("GradientBoostingClassifier", GradientBoostingClassifier(), parameters))

    # Multi-Layer Classifier
    parameters = {'hidden_layer_sizes':[(70,), (40, 40), (20, 20, 20)],
                  'activation':['logistic', 'relu'],
                  'batch_size':[128, 256],
                  'alpha':[0.001, 0.0001]}
    models.append(("MLPClassifier", MLPClassifier(), parameters))


    results = []
    names = []
    nb_cross_val = 2
    for name, model, parameter in models:
        print(name)
        clf = GridSearchCV(model, parameter, scoring='neg_log_loss', cv=nb_cross_val, n_jobs=4)
        clf.fit(X, y)
        res = clf.cv_results_

        with open(name+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for nb_set_params in range(len(res['params'])):
                res_str = '{} {} {}'.format(res['params'][nb_set_params],
                                            res['mean_test_score'][nb_set_params],
                                            res['std_test_score'][nb_set_params])
                writer.writerow([res_str])

    # for (i, name) in enumerate(names):
    #     print(name, results[i].keys())
