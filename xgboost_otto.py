#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn import preprocessing

#random_state = 42

########################################
# LOADING DATASETS IN PANDA DATAFRAMES #
########################################

train_csv_path = "./all/train.csv"
test_csv_path = "./all/test.csv"

train_df = pd.read_csv(train_csv_path,header=0)
test_df = pd.read_csv(test_csv_path,header=0)

# split df so that ids are isolated from features
# on drop la premiere colonne d'id, qui n'est pas un feature, cf. exemple tuto xgboost
train_data, train_target = train_df.iloc[:,1:-1], train_df.iloc[:,-1]
# pour TEST pas de classe (test_target) !!!: 
# on fait juste une prédiction de probabilité qu'on envoie sur Kaggle
test_data = test_df.iloc[:,1:]

# transformer les string de classe en etiquette int pour train
nbrRows = train_target.shape[0]
for rowIndex in range(nbrRows):
    train_target.at[rowIndex] = int(train_target.at[rowIndex][6])
    
pd.to_numeric(train_target)

###########################################
# TRAINING AND PREDICTION ON TEST DATASET #
###########################################

train_dmatrix = xgb.DMatrix(data=train_data,label=train_target)
xg_class = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree=0.3,learning_rate=0.05,max_depth=6,reg_alpha=5,n_estimators=100)

xg_class.fit(train_data,train_target)
preds = xg_class.predict_proba(test_data)

#####################################################
# CREATING .CSV RESPECTING KAGGLE SUBMISSION FORMAT #
#####################################################

preds = pd.DataFrame(preds)
namesRow = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
preds.columns = namesRow
preds.head()

preds.index +=1
preds.to_csv("results_xgboost.csv", encoding='utf-8',index=True,index_label="id")
