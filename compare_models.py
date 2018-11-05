import csv
import numpy as np
import warnings

from scipy.optimize import minimize

import xgboost as xgb
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, mean_squared_error

from toolbox import load_otto_db
from data_exploration import feature_importance

MODELS = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'MLPClassifier']


def compare_model(dir, models):
    """
    @brief: Load gridsearch result in .csv file
    @param:
            models: list of string containing the path to csv file
            dir: string, directory to files
     """

    for model in models:
        model_name = model.split('.')[0]
        best_params, best_mean_logloss, best_std_logloss = 0, -10000, 0

        with open(dir + model + '.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in reader:
                params = ' '.join(row[:-2])
                mean_logloss = float(row[-2])
                std_logloss = float(row[-1])

                if mean_logloss > best_mean_logloss - best_std_logloss:
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

            models: list of sklearn models (with params already passed)

    @return:
            print confusion matrix and log loss score for the final model (weighted prediction)

     """
    n_classes = len(np.unique(y_train))

    # Train all models on the training data, and print the resulting accuracy
    y_proba_pred = []
    for model_name, model in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(X_train, y_train)
            print('Score on model   => {}: {:.3f}'.format(model_name, model.score(X_val, y_val)))
            print('Logloss on model => {}: {:.3f}'.format(model_name, log_loss(y_val, model.predict_proba(X_val))))

            y_proba_pred.append(model.predict_proba(X_val))

    # We want to minimize the logloss
    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, y_proba_pred):
            final_prediction += weight * prediction
        return log_loss(y_val, final_prediction)

    ## Optimisation to find best weights between models

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
    print("   > Best Weights           : {}".format(optimal_weights))
    print("   > Initial ensemble Score : {}".format(log_loss_func(init_weights)))
    print("   > Final ensemble Score   : {}".format(res['fun']))

    # --------------------------------
    #  Find ensemble learning weights
    # --------------------------------

    y_pred_p = np.zeros((len(y_val), n_classes))

    for i_clf, clf in enumerate(models):
        y_pred_p += optimal_weights[i_clf] * clf[1].predict_proba(X_val)

    y_pred = np.argmax(y_pred_p, axis=1) + 1

    print(" * Accuracy on test set with ensemble learning : {:.4f}".format(accuracy_score(y_val,
                                                                                          y_pred)))
    print("{}".format(confusion_matrix(y_val, y_pred)))




if __name__ == '__main__':

    print('Comparing gridsearch on differents models')
    compare_model('data/', MODELS)

    X, y = load_otto_db()
    X_test = load_otto_db(test=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # RAM issue, remove if you have a better computer
    # X_train, X_val = X_train[:10000, :], X_val[:2000, :]
    # y_train, y_val = y_train[:10000], y_val[:2000]

    indices_best_features = feature_importance(X_train, y_train)

    models = []

    ## Best models according to gridsearch

    models.append(('XGBoost', xgb.XGBClassifier(objective='binary:logistic',
    colsample_bytree=0.8, learning_rate=0.1,
    max_depth=7, reg_alpha=0, n_estimators=100)))

    models.append(('Random Forest', RandomForestClassifier(max_depth=None, criterion='gini',
    n_estimators=100, min_samples_split=5)))

    models.append(('MLPClassifier', MLPClassifier(hidden_layer_sizes=(20, 20, 20), batch_size=256,
                                                  alpha=0.0001, activation='relu')))
    models.append(('Random Forest', LogisticRegression(C=10, penalty='l2')))


    print('MODEL MIX, ALL FEATURES')
    model_mix(X_train, y_train, X_val, y_val, models)

    # print('MODEL MIX, BEST FEATURES')
    # model_mix(X_train[:, indices_best_features], y_train,
    #           X_val[:, indices_best_features], y_val, models)
