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
from sklearn.calibration import CalibratedClassifierCV

from toolbox import load_otto_db

N_CROSS_VAL = 3
N_JOBS = 16
CSV_DIR = 'gridsearch_results'

if __name__ == "__main__":

    # Load data
    X, y = load_otto_db()
    models = []

    # Logistic Regression
    parameters = {'penalty':['l1', 'l2'],
                  'C':[0.1, 1., 10.],
                  'multi_class':['auto']}
    models.append(("LogisticRegression", LogisticRegression(), parameters))

    # RandomForestClassifier
    parameters = {'n_estimators':[10, 50, 100, 150],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 5, 10, 20],
                  'min_samples_split':[2, 4],
                  'n_jobs': [1]}
    models.append(("RandomForest", RandomForestClassifier(), parameters))

    # Multi-Layer Perceptron Classifier
    parameters = {'hidden_layer_sizes':[(90,), (50,), (40, 30), (50, 25, 15), (70, 50, 25, 15)],
                  'activation':['logistic', 'relu'],
                  'batch_size':[128, 256],
                  'alpha':[0.001, 0.0001],
                  'early_stopping':[True]}
    models.append(("MLPClassifier", MLPClassifier(), parameters))

    # DecisionTree
    parameters = {'splitter':['best', 'random'],
                  'criterion':['gini', 'entropy'],
                  'max_depth':[None, 10, 20, 50],
                  'min_samples_split':[2, 4, 6]}
    models.append(("DecisionTree", DecisionTreeClassifier(), parameters))

    # KNeighbors
    parameters = {'n_neighbors':[3, 5, 7],
                  'p':[1, 2],
                  'n_jobs': [1]}
    models.append(("KNeighbors", KNeighborsClassifier(), parameters))

    # Extreme Gradient Boosting classifier
    parameters = {'objective':['binary:logistic'],
                  'subsample':[0.7, 1],
                  'colsample_bytree':[0.8],
                  'learning_rate':[0.1],
                  'max_depth':[9],
                  'reg_alpha':[0,1],
                  'reg_lambda':[0,1],
                  'n_estimators':[100, 150],
                  'n_jobs': [-1]}
    models.append(("XGBoost", XGBClassifier(), parameters))

    # SVM : /!\ Very slow convergence
    # parameters = {'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    #               'C':[0.1, 1., 10., 100],
    #               'probability':[True],
    #               'gamma'=['scale']}
    # models.append(("SVC", SVC(), parameters))

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
