import numpy as np
import csv
import matplotlib.pyplot as plt
import warnings

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from toolbox import load_otto_db

N_CROSS_VAL = 4
N_JOBS = 8
CSV_DIR = 'gridsearch_results'

if __name__ == "__main__":

    # Load data
    X, y = load_otto_db()

    # Logistic Regression
    parameters = {'penalty':['l1', 'l2'],
                  'C':[0.1, 1., 10.],
                  'n_jobs': [N_JOBS]}
    models.append(("LogisticRegression", LogisticRegression(), parameters))

    # SVM
    parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
                  'C':[0.1, 1., 10., 100],
                  'probability':[True]}
    models.append(("SVC", SVC(), parameters))

    # KNeighbors
    parameters = {'n_neighbors':[3, 5, 7],
                  'p':[1, 2],
                  'n_jobs': [N_JOBS]}
    models.append(("KNeighbors", KNeighborsClassifier(), parameters))

    # DecisionTree
    parameters = {'splitter':['best', 'random'],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 10, 20, 50],
                  'min_samples_split':[2, 4, 6]}
    models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

    # RandomForestClassifier
    parameters = {'n_estimators':[10, 50, 100],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 5, 10],
                  'min_samples_split':[2, 4],
                  'n_jobs': [N_JOBS]}
    models.append(("RandomForest", RandomForestClassifier(), parameters))

    # GradientBoostingClassifier
    parameters = {'loss':['deviance', 'exponential'],
                  'n_estimators':[50, 100, 200],
                  'criterion':['friedman_mse', 'mae'],
                  'max_depth':[3, 7],
                  'min_samples_split':[2, 4],
                  'subsample':[0.7, 1.]}
    models.append(("GradientBoostingClassifier", GradientBoostingClassifier(), parameters))

    # Multi-Layer Perceptron Classifier
    parameters = {'hidden_layer_sizes':[(50,), (40, 30), (50, 25, 15)],
                  'activation':['logistic', 'relu'],
                  'batch_size':[128],
                  'alpha':[0.001, 0.0001],
                  'early_stopping' = [True]}
    models.append(("MLPClassifier", MLPClassifier(), parameters))

    # Extreme Gradient Boosting classifier
    parameters = {'objective':['binary:logistic'],
                  'subsample':[0.7, 1],
                  'colsample_bytree':[0.7, 1],
                  'learning_rate':[0.1],
                  'max_depth':[5, 7, 9],
                  'reg_alpha':[0,1],
                  'reg_lambda':[0,1],
                  'n_estimators':[100],
                  'n_jobs': [N_JOBS]}
    models.append(("XGBoost", XGBClassifier(), parameters))

    results = []
    names = []
    for name, model, parameter in models:
        print("\n==================================================")
        print("{:^50}".format(name))
        print("==================================================\n")

        # run grid search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            clf = GridSearchCV(model, parameter, scoring='neg_log_loss', cv=N_CROSS_VAL, n_jobs=N_JOBS, verbose=2)
            clf.fit(X, y)
            res = clf.cv_results_

        # save results to csv file
        with open('{}/{}.csv'.format(CSV_DIR, name), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for nb_set_params in range(len(res['params'])):
                res_str = '{} {} {}'.format(res['params'][nb_set_params],
                                            res['mean_test_score'][nb_set_params],
                                            res['std_test_score'][nb_set_params])
                writer.writerow([res_str])
