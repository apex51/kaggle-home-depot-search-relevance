import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

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

#################
# train/test split
x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

params = {}
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
iter_num = clf.best_iteration

print 'iteration number is {}'.format(iter_num)

# preds = clf.predict(deval)

# print np.sqrt(mean_squared_error(y_for_test, preds))
##################

params = {}
params['task'] = 'regression'
params['booster'] = 'gblinear'
params['objective'] = 'reg:linear'
params['eta'] = 1.0
params['alpha'] = 0.26
params['lambda'] = 3.3
params['lambda_bias'] = 0.8
params['silent'] = 1

num_rounds = 450

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

clf = xgb.train(params, dtrain, num_rounds)
preds = clf.predict(dtest)

df_preds = pd.read_csv(PROJECT_PATH + 'data/sample_submission.csv')
df_preds['relevance'] = preds
df_preds.loc[df_preds['relevance'] < 1, 'relevance'] = 1
df_preds.loc[df_preds['relevance'] > 3, 'relevance'] = 3
df_preds.to_csv(PROJECT_PATH + 'result/submission.csv', index=False)


#################
# use random forest

from sklearn.ensemble import RandomForestRegressor

clf = ExtraTreesRegressor(n_estimators = 500, n_jobs=-1, verbose=1, max_features=0.75)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)

df_preds = pd.read_csv(PROJECT_PATH + 'data/sample_submission.csv')
df_preds['relevance'] = preds
df_preds.loc[df_preds['relevance'] < 1, 'relevance'] = 1
df_preds.loc[df_preds['relevance'] > 3, 'relevance'] = 3
df_preds.to_csv(PROJECT_PATH + 'result/submission.csv', index=False)


#################
# use random forest classifier

from sklearn.ensemble import RandomForestClassifier

y_train[y_train == 2.75] = 2.67
y_train[y_train == 2.50] = 2.33
y_train[y_train == 2.25] = 2.33
y_train[y_train == 1.75] = 1.67
y_train[y_train == 1.50] = 1.33
y_train[y_train == 1.25] = 1.33

x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(x_for_train, y_for_train)
preds = clf.predict(x_for_test)

print np.sqrt(mean_squared_error(y_for_test, preds))

#################
# use poisson regression
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

x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

params = {}
params['objective'] = 'count:poisson'
params['booster'] = 'gblinear'
params['eta'] = 1.0
params['alpha'] = 0.26
params['lambda'] = 3.3
params['lambda_bias'] = 0.8
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
iter_num = clf.best_iteration

print 'iteration number is {}'.format(iter_num)

preds = clf.predict(deval)
preds = preds / 12.0


#################
# use xgb tree classifier

y_train[y_train == 2.75] = 2.67
y_train[y_train == 2.50] = 2.33
y_train[y_train == 2.25] = 2.33
y_train[y_train == 1.75] = 1.67
y_train[y_train == 1.50] = 1.33
y_train[y_train == 1.25] = 1.33

x_for_train, x_for_test, y_for_train, y_for_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

params = {}
params['booster'] = 'gbtree'
params['objective'] = 'multi:softmax'
params['colsample_bytree'] = 0.5
params['eta'] = 0.01
params['gamma'] = 1.3
params['max_depth'] = 15.0
params['min_child_weight'] = 9.0
params['subsample'] = 0.7
params['num_class'] = len(set(y_train))

num_rounds = 20000

dtrain = xgb.DMatrix(x_for_train, y_for_train)
deval = xgb.DMatrix(x_for_test, y_for_test)
watchlist = [(dtrain, 'train'), (deval, 'eval')]

clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
iter_num = clf.best_iteration

print 'iteration number is {}'.format(iter_num)

preds = clf.predict(deval, ntree_limit=iter_num)
print np.sqrt(mean_squared_error(y_for_test, preds))

