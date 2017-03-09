import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
# from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from datetime import datetime

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

#############################
# loading data
#############################

print '='*20
print 'loading features from pickles...'
start = datetime.now()

# load y_train from original features
# notice: original features did not drop the irrelevant features
with open(PROJECT_PATH + 'pickles/df_features.pkl') as f:
    df_train, df_test = pickle.load(f)

x_train = df_train.drop(labels=['relevance'], inplace=False, axis=1)
x_test = df_test
y_train = df_train['relevance']

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


#############################
# use hyperopt
#############################

def hyperopt_train_test(params):
    print '='*20
    print 'loading features from pickles...'
    start = datetime.now()

    # k-fold
    kf = KFold(74067, n_folds=5, shuffle=True, random_state=22)

    n_rounds = []
    scores = []

    for train_index, test_index in kf:
        x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
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
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'lambda': hp.quniform('lambda', 0, 5, 0.05),
    'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
    'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
    'silent': 1
}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=30, trials=trials)


# 'alpha': [0.26],
# 'eta': [1.0],
# 'lambda': [3.3000000000000003],
# 'lambda_bias': [0.8]},

# 'loss': 0.472701,
# 'eval_time': '0:01:57.496057',
# 'n_rounds': [359, 464, 539, 493, 389], mean = 450
# 'scores': [0.471436, 0.470615, 0.476284, 0.475131, 0.470039]},








