import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# project path
PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'
PROJECT_PATH_LINUX = '/home/jiangh/home_depot/'
model_types_rgrs = ['xgb_tree_rgrs', 'xgb_linear_rgrs', 'rf_rgrs', 'et_rgrs', 'lasso_rgrs', 'ridge_rgrs', 'keras_rgrs']
model_types_clf = ['rf_clf', 'xgb_tree_clf']

######################################
# copy keras level-1 result from cuda machine
######################################

import os
cmd = 'scp jiangh@192.168.204.249:{}pickles/level1_keras_rgrs.pkl {}pickles/'.format(PROJECT_PATH_LINUX, PROJECT_PATH)
os.system(cmd)

######################################
# loading original y_train
######################################

with open(PROJECT_PATH + 'pickles/df_features.pkl') as f:
    df_train, df_test = pickle.load(f)

y_train = df_train['relevance'].copy()

######################################
# loading stratified 5-fold split
######################################

with open(PROJECT_PATH + 'pickles/df_preprocessed_sk5f.pkl') as f:
    sk5f = pickle.load(f)

######################################
# test each result (no need to run again because result's asured)
######################################

# y_reindexed = []
# for train_index, test_index in sk5f:
#     _, y_test_sp = y_train[train_index], y_train[test_index]
#     y_reindexed.append(y_test_sp)
# y_reindexed = np.concatenate(y_reindexed, axis=0)

# for type_name in model_types_rgrs:
#     with open(PROJECT_PATH + 'pickles/level1_{}.pkl'.format(type_name)) as f:
#         result_5folds, preds = pickle.load(f)

#     score = np.sqrt(mean_squared_error(y_reindexed, result_5folds))
#     print '{}\'s score is {}'.format(type_name, score)

######################################
# loading level1 results
######################################

x_train = []
x_test = []

# loading rgrs results
for type_name in model_types_rgrs:
    with open(PROJECT_PATH + 'pickles/level1_{}.pkl'.format(type_name)) as f:
        result_5folds, preds = pickle.load(f)
    x_train.append(result_5folds.flatten())
    x_test.append(preds.flatten())

x_train = np.array(x_train).T
x_test = np.array(x_test).T

# loading clf results
for type_name in model_types_clf:
    with open(PROJECT_PATH + 'pickles/level1_{}.pkl'.format(type_name)) as f:
        result_5folds, preds = pickle.load(f)
        x_train = np.concatenate((x_train, result_5folds), axis=1)
        x_test = np.concatenate((x_test, preds), axis=1)

print 'x_train\'s shape: {}'.format(x_train.shape)
print 'x_test\'s shape: {}'.format(x_test.shape)

y_train_tmp = []
for train_index, test_index in sk5f:
    _, y_test_sp = y_train[train_index], y_train[test_index]
    y_train_tmp.append(y_test_sp)
y_train = np.concatenate(y_train_tmp, axis=0)

print 'y_train\'s length: {}'.format(len(y_train))

######################################
# train rgrs
######################################

# use xgb tree rgrs to test results
import xgboost as xgb
from sklearn.cross_validation import train_test_split

x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

params = {}
params['booster'] = 'gbtree'
params['objective'] = 'reg:linear'
params['eta'] = 0.01
# params['subsample'] = 0.7
# params['colsample_bytree'] = 0.7
# params['max_depth'] = 10
# params['min_child_weight'] = 5
params['silent'] = 1

num_rounds = 20000

dtrain = xgb.DMatrix(x_for_train, y_for_train)
deval = xgb.DMatrix(x_for_test, y_for_test)
watchlist = [(dtrain, 'train'), (deval, 'eval')]

clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
# 0.446973

#########################################

# use xgb poisson rgrs to test results
import xgboost as xgb
from sklearn.cross_validation import train_test_split

y_train_poisson = [int(round(x * 12)) for x in y_train]
x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train_poisson, test_size=0.2, random_state=22)

params = {}
params['booster'] = 'gbtree'
params['objective'] = 'count:poisson'
params['eta'] = 0.01
# params['subsample'] = 0.7
# params['colsample_bytree'] = 0.7
# params['max_depth'] = 10
# params['min_child_weight'] = 5
params['silent'] = 1

num_rounds = 20000

dtrain = xgb.DMatrix(x_for_train, y_for_train)
deval = xgb.DMatrix(x_for_test, y_for_test)
watchlist = [(dtrain, 'train'), (deval, 'eval')]

clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
preds = clf.predict(deval)
score = np.sqrt(mean_squared_error(y_for_test/12.0, preds/12.0))

# 0.4473253497763896

#########################################

# use random forest rgrs to test results
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

rgrs = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rgrs.fit(x_for_train, y_for_train)
preds = rgrs.predict(x_for_test)

score = np.sqrt(mean_squared_error(y_for_test, preds))
# 0.4499765017013394

######################################
# generate results
######################################

params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.01
# params['subsample'] = 0.7
# params['colsample_bytree'] = 0.7
# params['max_depth'] = 10
# params['min_child_weight'] = 5
params['silent'] = 1

num_rounds = 562

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

clf = xgb.train(params, dtrain, num_rounds)
preds = clf.predict(dtest)

df_preds = pd.read_csv(PROJECT_PATH + 'data/sample_submission.csv')
df_preds['relevance'] = preds
df_preds.loc[df_preds['relevance'] < 1, 'relevance'] = 1
df_preds.loc[df_preds['relevance'] > 3, 'relevance'] = 3
df_preds.to_csv(PROJECT_PATH + 'result/submission.csv', index=False)

######################################
# parameter: xgb tree rgrs
######################################

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def hyperopt_train_test(params):
    print '='*20
    print 'loading features from pickles...'
    start = datetime.now()

    # k-fold
    kf = KFold(74067, n_folds=5, shuffle=True, random_state=22)

    n_rounds = []
    scores = []

    for train_index, test_index in kf:
        x_train_sp, x_test_sp = x_train[train_index], x_train[test_index]
        y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]
        
        dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
        deval = xgb.DMatrix(x_test_sp, y_test_sp)
        watchlist = [(dtrain, 'train'), (deval, 'eval')]
        num_rounds = 20000
        
        clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
        n_rounds.append(clf.best_iteration)
        scores.append(clf.best_score)

    eval_time = datetime.now() - start
    print 'done, {}'.format(eval_time)

    return {'loss': np.mean(scores),
            'n_rounds': n_rounds,
            'scores': scores,
            'eval_time': str(eval_time)}

def f(params):
    results = hyperopt_train_test(params)
    return {'loss': results['loss'],
            'status': STATUS_OK,
            'other_stuff': {'n_rounds': results['n_rounds'],
                            'scores': results['scores'],
                            'eval_time': results['eval_time']
                            }
            }

space = {
    'task': 'regression',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 0.05, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
    'silent': 1
}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=30, trials=trials)
