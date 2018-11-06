import csv
import numpy as np
import warnings
from scipy.optimize import minimize

import pandas as pd
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, mean_squared_error
from xgboost import XGBClassifier

from toolbox import load_otto_db
from data_exploration import feature_importance


COMPARE_MODELS = True
MODELS = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'MLPClassifier', 'XGBoost']
GRIDSEARCH_DIR = 'gridsearch_results'

N_JOBS = -1
LIMITED_RAM = False
OUTPUT_DIR = 'data'


def compare_model(dir, models):
    """
    @brief: Load gridsearch result in .csv file
    @param:
            models: list of string containing the path to csv file
            dir: string, directory to files
     """

    for model in models:
        model_name = model.split('.')[0]
        best_params, best_mean_logloss, best_std_logloss = {}, -10000, 1

        with open('{}/{}.csv'.format(dir, model)) as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in reader:
                params = ' '.join(row[:-2])
                mean_logloss = float(row[-2])
                std_logloss = float(row[-1])

                if mean_logloss + std_logloss > best_mean_logloss + best_std_logloss:
                    best_mean_logloss = mean_logloss
                    best_std_logloss = std_logloss
                    best_params = params

        print('Model: {} => neg_log_loss: {}, std: {}, params: {}\n'.format(model_name,
                                                                            best_mean_logloss,
                                                                            best_std_logloss,
                                                                            best_params))


def model_mix(X_train, y_train, X_val, y_val, models):
    """
    @brief: Load gridsearch result in .csv file
    @param:
            X_train: ndarray (n_samples, n_features), array of samples to train
            y_train: ndarray (n_samples,), array of targets for each train sample

            X_val: ndarray (n_samples, n_features), array of samples to test
            y_val: ndarray (n_samples,), array of targets for each test sample

            models: list of sklearn models (with params already passed) as a
                    list of tuple (name, model)

    @return:
            (print confusion matrix and log loss score for the final model (weighted prediction))
            models: list of learned sklearn models as a list of tuple (name, model)
            optimal_weights: weights for ponderation between model prediction,
                             optimized for those models
     """
    n_classes = len(np.unique(y_train))

    # --------------------------------
    #  Train each model independently
    # --------------------------------

    # Train all models on the training data, and print the resulting accuracy
    y_proba_pred = []
    for i_model, model in enumerate(models):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model_name = models[i_model][0]
            print('Training model {}'.format(model_name))
            models[i_model] = (model_name, CalibratedClassifierCV(models[i_model][1], method='isotonic', cv=3))
            model = models[i_model][1]
            model.fit(X_train, y_train)
            print(' * Score on model   : {:.3f}'.format(model.score(X_val, y_val)))
            print(' * Logloss on model : {:.3f}'.format(log_loss(y_val, model.predict_proba(X_val))))
            y_proba_pred.append(model.predict_proba(X_val))

    # --------------------------------
    #  Find ensemble learning weights
    # --------------------------------

    # We want to minimize the logloss
    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, y_proba_pred):
            final_prediction += weight * prediction
        return log_loss(y_val, final_prediction)

    # Uniform initialisation
    init_weights = np.ones((len(y_proba_pred),)) / len(y_proba_pred)
    # init_weights = np.array([0.5, 0.4, 0.1])
    # We want to have the weight at 1
    constraint = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(y_proba_pred)
    # Compute best weights (method chosen with the advive of Kaggle kernel)
    res = minimize(log_loss_func, init_weights, method='SLSQP', bounds=bounds,
                   constraints=constraint)
    optimal_weights = res['x']

    # print results
    print("Ensamble learning results :")
    print(" * Best Weights           : {}".format(optimal_weights))
    print(" * Initial ensemble Score : {}".format(log_loss_func(init_weights)))
    print(" * Final ensemble Score   : {}".format(res['fun']))

    y_val_pred_p = model_mix_predict(X_val, models, optimal_weights, n_classes)
    y_val_pred = np.argmax(y_val_pred_p, axis=1) + 1

    print(" * Accuracy on validation set with ensemble learning : {:.4f}".format(accuracy_score(y_val, y_val_pred)))
    print("{}".format(confusion_matrix(y_val, y_val_pred)))

    return models, optimal_weights


def model_mix_predict(X, models, optimal_weights, n_classes):
    """
    @brief: take a list of sklearn models, weights and a dataset and return the weighted prediction
            over the samples

    @param:
            X: ndarray, (n_samples, n_features), dataset to predict
            models: list of tuple (name, model), with model a sklearn model already trained
            optimal_weights: list of float, weight for each model (sum(weight)==1)

    @return:
            y_pred_p: ndarray, (n_samples, n_classes), probability for each class for each sample
    """
    y_pred_p = np.zeros((X.shape[0], n_classes))
    for i_model, model in enumerate(models):
        y_pred_p += optimal_weights[i_model] * model[1].predict_proba(X)
    return y_pred_p


if __name__ == '__main__':

    if COMPARE_MODELS:
        print("\n==================================================")
        print("{:^50}".format("COMPARING GRIDSEARCH ON DIFFERENTS MODELS"))
        print("==================================================")
        compare_model(GRIDSEARCH_DIR, MODELS)


    print("\n==================================================")
    print("{:^50}".format("LOADING DATASET"))
    print("==================================================")

    print("Parsing csv files...")
    X, y = load_otto_db()
    X_test = load_otto_db(test=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    n_classes = len(np.unique(y))

    # RAM issue, remove if you have a better computer
    if LIMITED_RAM:
        print('WARNING : selecting partial dataset for RAM saving')
        X_train, X_val = X_train[:10000, :], X_val[:2000, :]
        y_train, y_val = y_train[:10000], y_val[:2000]

    print("Dimensions of datasets :")
    print(" * Training set   : {}".format(X_train.shape))
    print(" * Validation set : {}".format(X_val.shape))
    print(" * Test set       : {}".format(X_test.shape))


    # print("\n==================================================")
    # print("{:^50}".format("FEATURES IMPORTANCE"))
    # print("==================================================")
    # indices_best_features = feature_importance(X_train, y_train)
    # print(indices_best_features)


    print("\n==================================================")
    print("{:^50}".format("TRAINING MODELS"))
    print("==================================================")
    models = []

    ## Best models according to gridsearch
    parameters = {'objective': 'binary:logistic',
                  'n_estimators': 150,
                  'max_depth': 9,
                  'learning_rate': 0.1,
                  'subsample': 0.7,
                  'colsample_bytree': 0.8,
                  'reg_lambda': 0,
                  'reg_alpha': 1,
                  'n_jobs': N_JOBS}
    models.append(('XGBoost 1', XGBClassifier(**parameters)))

    parameters = {'objective': 'binary:logistic',
                  'n_estimators': 150,
                  'max_depth': 9,
                  'learning_rate': 0.1,
                  'subsample': 0.7,
                  'colsample_bytree': 0.8,
                  'reg_lambda': 1,
                  'reg_alpha': 0,
                  'n_jobs': N_JOBS}
    models.append(('XGBoost 2', XGBClassifier(**parameters)))

    parameters = {'criterion': 'gini',
                  'n_estimators': 150,
                  'max_depth': None,
                  'min_samples_split': 4,
                  'n_jobs': N_JOBS}
    models.append(('Random Forest 1', RandomForestClassifier(**parameters)))

    parameters = {'criterion': 'gini',
                  'n_estimators': 200,
                  'max_depth': None,
                  'min_samples_split': 4,
                  'n_jobs': N_JOBS}
    models.append(('Random Forest 2', RandomForestClassifier(**parameters)))

    parameters = {'hidden_layer_sizes': (90,),
                  'activation': 'logistic',
                  'alpha': 0.001,
                  'early_stopping': True,
                  'batch_size': 128}
    models.append(('MLP Classifier 1', MLPClassifier(**parameters)))

    parameters = {'hidden_layer_sizes': (120,),
                  'activation': 'logistic',
                  'alpha': 0.001,
                  'early_stopping': True,
                  'batch_size': 128}
    models.append(('MLP Classifier 2', MLPClassifier(**parameters)))

    parameters = {'C': 10.0,
                  'penalty': 'l2'}
    models.append(('Logistic Regression', LogisticRegression(**parameters)))


    print('\nMODEL MIX, ALL FEATURES\n')
    trained_models, optimal_weights = model_mix(X_train, y_train, X_val, y_val, models)

    # print('\nMODEL MIX, BEST FEATURES\n')
    # model_mix(X_train[:, indices_best_features], y_train,
    #           X_val[:, indices_best_features], y_val, models)

    print("\n==================================================")
    print("{:^50}".format("FINAL PREDICTIONS"))
    print("==================================================")

    print("Running final predictions on test set...")
    y_test_pred = model_mix_predict(X_test, trained_models, optimal_weights, n_classes)

    print("Saving predictions to .csv file respecting kaggle submission format...")
    preds = pd.DataFrame(y_test_pred)
    namesRow = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6",
                "Class_7", "Class_8", "Class_9"]
    preds.columns = namesRow
    preds.head()
    preds.index += 1
    preds.to_csv(OUTPUT_DIR + "/results_model_mix.csv", encoding='utf-8', index=True, index_label="id")
