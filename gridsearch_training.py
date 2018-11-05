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
from sklearn.model_selection import GridSearchCV

from toolbox import load_otto_db


if __name__ == "__main__":

    # Load data
    X, y = load_otto_db()

    # Logistic Regression
    parameters = {'penalty':['l1', 'l2'],
                  'C':[0.1, 1., 10.]}
    models.append(("LogisticRegression", LogisticRegression(), parameters))

    # SVM
    parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                  'C':[0.1, 1., 10., 100],
                  'probability':True}
    models.append(("SVC", SVC(), parameters))

    # KNeighbors
    parameters = {'n_neighbors':[3, 5, 7],
                  'leaf_size':[10, 20, 30, 40],
                  'p':[1, 2]}
    models.append(("KNeighbors", KNeighborsClassifier(), parameters))

    # DecisionTree
    parameters = {'splitter':['best', 'random'],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[3, 5, 10, 20],
                  'min_samples_split':[2, 3, 5]}
    models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

    # RandomForestClassifier
    parameters = {'n_estimators':[10, 50, 100],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 3, 10],
                  'min_samples_split':[2, 3, 5]}
    models.append(("RandomForest", RandomForestClassifier(), parameters))

    # GradientBoostingClassifier
    parameters = {'loss':['deviance', 'exponential'],
                  'n_estimators':[10, 50, 100],
                  'criterion':['friedman_mse', 'mae'],
                  'max_depth':[3, 10],
                  'min_samples_split':[2, 3, 5]}
    models.append(("GradientBoostingClassifier", GradientBoostingClassifier(), parameters))

    # Multi-Layer Perceptron Classifier
    parameters = {'hidden_layer_sizes':[(70,), (40, 40), (20, 20, 20)],
                  'activation':['logistic', 'relu'],
                  'batch_size':[128],
                  'alpha':[0.001, 0.0001]}
    models.append(("MLPClassifier", MLPClassifier(), parameters))


    results = []
    names = []
    nb_cross_val = 4
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
