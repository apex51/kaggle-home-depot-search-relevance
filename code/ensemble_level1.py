import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

# mode type: rgrs OR clf
mode = 'clf'

# check on which machine this program runs
# Darwin: my mbp
# Linux: cuda machine
import os
sys_name = os.uname()[0]

if sys_name == 'Darwin':
    PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'
if sys_name == 'Linux':
    PROJECT_PATH = '/home/jiangh/home_depot/'

######################################
# loading data
######################################

print '='*20
print 'loading features from pickles...'
start = datetime.now()

# load y_train from original features
# notice: original features did not drop the irrelevant features
with open(PROJECT_PATH + 'pickles/df_features.pkl') as f:
    df_train, df_test = pickle.load(f)

x_train = df_train.drop(labels=['relevance'], inplace=False, axis=1)
x_test = df_test
y_train = df_train['relevance'].copy()

# map some minor results
if mode == 'clf':
    y_train[y_train == 2.75] = 2.67
    y_train[y_train == 2.50] = 2.33
    y_train[y_train == 2.25] = 2.33
    y_train[y_train == 1.75] = 1.67
    y_train[y_train == 1.50] = 1.33
    y_train[y_train == 1.25] = 1.33
    y_train = y_train.apply(str)
if mode == 'poisson':
    y_train[y_train == 3.00] = 36
    y_train[y_train == 2.75] = 33
    y_train[y_train == 2.67] = 32
    y_train[y_train == 2.50] = 30
    y_train[y_train == 2.33] = 28
    y_train[y_train == 2.25] = 27
    y_train[y_train == 2.00] = 24
    y_train[y_train == 1.75] = 21
    y_train[y_train == 1.67] = 20
    y_train[y_train == 1.50] = 18
    y_train[y_train == 1.33] = 16
    y_train[y_train == 1.25] = 15
    y_train[y_train == 1.00] = 12


# load data from unigram features
with open(PROJECT_PATH + 'pickles/df_features_unigram.pkl') as f:
    df_train_unigram, df_test_unigram = pickle.load(f)

# load data from bigram features
with open(PROJECT_PATH + 'pickles/df_features_bigram.pkl') as f:
    df_train_bigram, df_test_bigram = pickle.load(f)

# load data from trigram features
with open(PROJECT_PATH + 'pickles/df_features_trigram.pkl') as f:
    df_train_trigram, df_test_trigram = pickle.load(f)

# load data from svd features
with open(PROJECT_PATH + 'pickles/df_features_svd.pkl') as f:
    df_train_svd, df_test_svd = pickle.load(f)

x_train = pd.concat((x_train, df_train_unigram, df_train_bigram, df_train_trigram, df_train_svd), axis=1)
x_test = pd.concat((x_test, df_test_unigram, df_test_bigram, df_test_trigram, df_test_svd), axis=1)

print 'done, {}'.format(datetime.now() - start)

######################################
# loading stratified 5-fold split
######################################

with open(PROJECT_PATH + 'pickles/df_preprocessed_sk5f.pkl') as f:
    sk5f = pickle.load(f)

######################################
# training each model
######################################

if sys_name == 'Darwin':
    if mode == 'rgrs':
        model_types = ['xgb_tree_rgrs', 'xgb_linear_rgrs', 'rf_rgrs', 'et_rgrs', 'lasso_rgrs', 'ridge_rgrs']
    if mode == 'clf':
        # model_types = ['rf_clf', 'xgb_tree_clf']
        model_types = ['xgb_tree_clf']
    if mode == 'poisson':
        model_types = ['xgb_poisson']
if sys_name == 'Linux':
    model_types = ['keras_rgrs']

# train each type in model_types
for type_name in model_types:
    # for xgb_tree_rgrs
    if type_name == 'xgb_tree_rgrs':
        # params
        params = {}
        params['task'] = 'regression'
        params['booster'] = 'gbtree'
        params['objective'] = 'reg:linear'
        params['colsample_bytree'] = 0.5
        params['eta'] = 0.01
        params['gamma'] = 1.3
        params['max_depth'] = 15.0
        params['min_child_weight'] = 9.0
        params['subsample'] = 0.7
        num_rounds = 990
        # num_rounds = 1

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]
            dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
            dtest = xgb.DMatrix(x_test_sp)
            clf = xgb.train(params, dtrain, num_rounds)
            preds = clf.predict(dtest)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)
        clf = xgb.train(params, dtrain, num_rounds)
        preds = clf.predict(dtest)

        with open(PROJECT_PATH + 'pickles/level1_xgb_tree_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)
    
    # for xgb_linear_rgrs
    if type_name == 'xgb_linear_rgrs':
        # params
        params = {}
        params['task'] = 'regression'
        params['booster'] = 'gblinear'
        params['objective'] = 'reg:linear'
        params['eta'] = 1.0
        params['alpha'] = 0.26
        params['lambda'] = 3.30
        params['lambda_bias'] = 0.8
        num_rounds = 450
        # num_rounds = 1

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]
            dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
            dtest = xgb.DMatrix(x_test_sp)
            clf = xgb.train(params, dtrain, num_rounds)
            preds = clf.predict(dtest)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)
        clf = xgb.train(params, dtrain, num_rounds)
        preds = clf.predict(dtest)

        with open(PROJECT_PATH + 'pickles/level1_xgb_linear_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)
    
    # for rf_rgrs
    if type_name == 'rf_rgrs':
        # params
        params = {}
        params['n_estimators'] = 1000
        # params['n_estimators'] = 1
        params['max_features'] = 0.65
        params['n_jobs'] = -1
        params['random_state'] = 22

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            clf = RandomForestRegressor(**params)
            clf.fit(x_train_sp, y_train_sp)
            preds = clf.predict(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        clf = RandomForestRegressor(**params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        with open(PROJECT_PATH + 'pickles/level1_rf_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for et_rgrs
    if type_name == 'et_rgrs':
        # params
        params = {}
        params['n_estimators'] = 1000
        # params['n_estimators'] = 1
        params['max_features'] = 0.75
        params['n_jobs'] = -1
        params['random_state'] = 22

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            clf = ExtraTreesRegressor(**params)
            clf.fit(x_train_sp, y_train_sp)
            preds = clf.predict(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        clf = ExtraTreesRegressor(**params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        with open(PROJECT_PATH + 'pickles/level1_et_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for lasso_rgrs
    if type_name == 'lasso_rgrs':
        # params
        params = {}
        params['alpha'] = 2.6119839093556226e-05

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            clf = Lasso(**params)
            clf.fit(x_train_sp, y_train_sp)
            preds = clf.predict(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        clf = Lasso(**params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        with open(PROJECT_PATH + 'pickles/level1_lasso_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for ridge_rgrs
    if type_name == 'ridge_rgrs':
        # params
        params = {}
        params['alpha'] = 0.7907994420477394

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            clf = Ridge(**params)
            clf.fit(x_train_sp, y_train_sp)
            preds = clf.predict(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        clf = Ridge(**params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        with open(PROJECT_PATH + 'pickles/level1_ridge_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for keras_rgrs
    if type_name == 'keras_rgrs':
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.normalization import BatchNormalization
        from keras.layers.advanced_activations import PReLU
        from keras.utils import np_utils, generic_utils

        # params
        params = {}
        params['activation'] = 'sigmoid'
        params['batch_size'] = 64
        params['hidden_dropout_first'] = 0.5
        params['hidden_dropout_second'] = 0.2
        params['hidden_units_first'] = 128
        params['hidden_units_second'] = 256
        params['nb_epoch'] = 50

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            model = Sequential()
            # first layer
            model.add(Dense(params["hidden_units_first"], input_dim=x_train.shape[1], init='glorot_uniform', activation=params['activation']))
            model.add(Dropout(params["hidden_dropout_first"]))
            # second layer
            model.add(Dense(params["hidden_units_second"], init='glorot_uniform', activation=params['activation']))
            model.add(Dropout(params["hidden_dropout_second"]))
            # output layer
            model.add(Dense(1, init='glorot_uniform', activation='linear'))
            model.compile(loss='mean_squared_error', optimizer="adam")

            scaler = StandardScaler()
            x_train_sp = scaler.fit_transform(x_train_sp)
            x_test_sp = scaler.transform(x_test_sp)

            model.fit(x_train_sp, y_train_sp, nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], validation_split=0)
            preds = model.predict(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        model = Sequential()
        # first layer
        model.add(Dense(params["hidden_units_first"], input_dim=x_train.shape[1], init='glorot_uniform', activation=params['activation']))
        model.add(Dropout(params["hidden_dropout_first"]))
        # second layer
        model.add(Dense(params["hidden_units_second"], init='glorot_uniform', activation=params['activation']))
        model.add(Dropout(params["hidden_dropout_second"]))
        # output layer
        model.add(Dense(1, init='glorot_uniform', activation='linear'))
        model.compile(loss='mean_squared_error', optimizer="adam")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit(x_train, y_train, nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], validation_split=0)
        preds = model.predict(x_test)

        with open(PROJECT_PATH + 'pickles/level1_keras_rgrs.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for rf_clf
    if type_name == 'rf_clf':
        # params
        params = {}
        params['n_estimators'] = 1000
        # params['n_estimators'] = 1
        params['max_features'] = 0.65
        params['n_jobs'] = -1
        params['random_state'] = 22

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

            clf = RandomForestClassifier(**params)
            clf.fit(x_train_sp, y_train_sp)
            preds = clf.predict_proba(x_test_sp)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        clf = RandomForestClassifier(**params)
        clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_test)

        with open(PROJECT_PATH + 'pickles/level1_rf_clf.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)

    # for xgb_poisson
    if type_name == 'xgb_poisson':
        # params
        params = {}
        params['objective'] = 'count:poisson'
        params['eta'] = 0.01
        # params['subsample'] = 0.7
        # params['colsample_bytree'] = 0.7
        # params['max_depth'] = 10
        # params['min_child_weight'] = 5
        params['silent'] = 1
        num_rounds = 450
        # num_rounds = 1

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]
            dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
            dtest = xgb.DMatrix(x_test_sp)
            clf = xgb.train(params, dtrain, num_rounds)
            preds = clf.predict(dtest)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_test)
        clf = xgb.train(params, dtrain, num_rounds)
        preds = clf.predict(dtest)

        with open(PROJECT_PATH + 'pickles/level1_xgb_poisson.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)
    
    # for xgb_poisson
    if type_name == 'xgb_tree_clf':
        y_train_float = y_train.apply(float)

        # params
        params = {}
        params['booster'] = 'gbtree'
        params['objective'] = 'multi:softprob'
        params['colsample_bytree'] = 0.5
        params['eta'] = 0.01
        params['gamma'] = 1.3
        params['max_depth'] = 15.0
        params['min_child_weight'] = 9.0
        params['subsample'] = 0.7
        params['num_class'] = len(set(y_train_float))
        num_rounds = 13

        # recorder
        result_5folds = []
        # split into two parts, calculate result for each fold
        for train_index, test_index in sk5f:
            x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_sp, y_test_sp = y_train_float[train_index], y_train_float[test_index]
            dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
            dtest = xgb.DMatrix(x_test_sp)
            clf = xgb.train(params, dtrain, num_rounds)
            preds = clf.predict(dtest)
            result_5folds.append(preds)

        result_5folds = np.concatenate(result_5folds, axis=0)
        dtrain = xgb.DMatrix(x_train, y_train_float)
        dtest = xgb.DMatrix(x_test)
        clf = xgb.train(params, dtrain, num_rounds)
        preds = clf.predict(dtest)

        with open(PROJECT_PATH + 'pickles/level1_xgb_tree_clf.pkl', 'wb') as f:
            pickle.dump((result_5folds, preds), f)













