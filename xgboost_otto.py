#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn import preprocessing

from toolbox import load_otto_db


#################
# LOAD DATASETS #
#################

X_train, y_train = load_otto_db()
X_test = load_otto_db(test=True)

print("Dimensions of datasets :")
print(" * Training set : {}".format(X.shape))
print(" * Test set     : {}".format(X_test.shape))


###########################################
# TRAINING AND PREDICTION ON TEST DATASET #
###########################################

xgb_clf = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree=0.3,learning_rate=0.05,max_depth=6,reg_alpha=5,n_estimators=100)

xgb_clf.fit(X_train, y_train)
y_test_pred = xgb_clf.predict_proba(X_test)


#####################################################
# CREATING .CSV RESPECTING KAGGLE SUBMISSION FORMAT #
#####################################################

preds = pd.DataFrame(y_test_pred)
namesRow = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
preds.columns = namesRow
preds.head()

preds.index +=1
preds.to_csv("results_xgboost.csv", encoding='utf-8',index=True,index_label="id")
