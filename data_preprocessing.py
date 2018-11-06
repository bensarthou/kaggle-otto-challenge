#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from time import time
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from toolbox import load_otto_db

##############
# PARAMETERS #
##############

#BaseClf = XGBClassifier
BaseClf = RandomForestClassifier

# parameters for XGBoost classifier
# parameters = {'objective': 'binary:logistic',
#               'n_estimators': 100,
#               'max_depth': 9,
#               'subsample': 0.7,
#               'colsample_bytree': 0.8,
#               'learning_rate': 0.1,
#               'reg_alpha': 0,
#               'reg_lambda': 1,
#               'n_jobs': -1}

# parameters for RandomForestClassifier
parameters = {'n_estimators': 100,
              'n_jobs': -1}

#############
# FUNCTIONS #
#############

def fit_model_and_print_results(model, X_train, y_train, X_test, y_test, title):
    # train model
    begin = time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = model.fit(X_train, y_train)
    duration = time() - begin
    # make predictions
    y_test_preds_p = model.predict_proba(X_test)
    y_test_preds = model.predict(X_test)
    # display results
    print("\n{} :".format(title))
    print(" * training time = {:.1f}s".format(duration))
    print(" * logloss       = {:.4f}".format(log_loss(y_test, y_test_preds_p, eps=1e-15, normalize=True)))
    print(" * accuracy      = {:.4f}".format(accuracy_score(y_test, y_test_preds)))


################################################################################
if __name__ == "__main__":

    print("\n==================================================")
    print("{:^50}".format("LOAD DATASETS"))
    print("==================================================\n")

    X, y = load_otto_db()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=36)

    print("Dimensions of datasets :")
    print(" * Training set : {}".format(X_train.shape))
    print(" * Test set     : {}".format(X_test.shape))


    ############################################################################
    if False:

        print("\n==================================================")
        print("{:^50}".format("FEATURES NORMALIZATION"))
        print("==================================================")

        # Without normalization
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train, y_train, X_test, y_test, 'WITHOUT normalization')

        # With features normalization
        clf = BaseClf(**parameters)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        fit_model_and_print_results(clf, X_train_scaled, y_train, X_test_scaled, y_test, 'WITH Standard normalization')

        # With features normalization
        clf = BaseClf(**parameters)
        scaler = MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        fit_model_and_print_results(clf, X_train_scaled, y_train, X_test_scaled, y_test, 'WITH MinMax normalization')


    ############################################################################
    if False:

        print("\n==================================================")
        print("{:^50}".format("FEATURES SELECTION BY RFE"))
        print("==================================================")

        # All features
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train, y_train, X_test, y_test, 'with ALL features')

        # Selected features
        N_COMPONENTS = 30
        clf = BaseClf(**parameters)
        selector = RFE(clf, n_features_to_select=N_COMPONENTS, step=2)
        X_train_rfe = selector.fit_transform(X_train, y_train)
        X_test_rfe = selector.transform(X_test)
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train_rfe, y_train, X_test_rfe, y_test, 'with {} RFE features'.format(N_COMPONENTS))


    ############################################################################
    if False:

        print("\n==================================================")
        print("{:^50}".format("DIMENSIONNALITY REDUCTION BY PCA"))
        print("==================================================")

        # All features
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train, y_train, X_test, y_test, 'with ALL features')

        # PCA features
        N_COMPONENTS = 60
        # normalize features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # reduce dimensions
        pca = PCA(n_components=N_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        # run model
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train_pca, y_train, X_test_pca, y_test, 'with {} PCA features'.format(N_COMPONENTS))


    ############################################################################
    if False:

        print("\n==================================================")
        print("{:^50}".format("DIMENSIONNALITY REDUCTION BY TRUNCATED SVD"))
        print("==================================================")

        # All features
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train, y_train, X_test, y_test, 'with ALL features')

        # PCA features
        N_COMPONENTS = 92
        # normalize features
        # scaler = StandardScaler().fit(X_train)
        # X_train_scaled = scaler.transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        # reduce dimensions
        reducer = TruncatedSVD(n_components=N_COMPONENTS)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        # run model
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train_reduced, y_train, X_test_reduced, y_test, 'with {} TruncatedSVD features'.format(N_COMPONENTS))


    ############################################################################
    if False:

        print("\n==================================================")
        print("{:^50}".format("PREDICTIONS CALIBRATION"))
        print("==================================================")

        # No calibration
        clf = BaseClf(**parameters)
        fit_model_and_print_results(clf, X_train, y_train, X_test, y_test, 'WITHOUT calibration')

        # With Calibration
        clf = BaseClf(**parameters)
        calib_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        fit_model_and_print_results(calib_clf, X_train, y_train, X_test, y_test, 'WITH calibration')
