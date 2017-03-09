import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


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

    t = params['type']
    del params['type']

    for train_index, test_index in kf:
        x_train_sp, x_test_sp = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_sp, y_test_sp = y_train[train_index], y_train[test_index]

        if t == 'svr':
            scaler = StandardScaler()
            x_train_sp = scaler.fit_transform(x_train_sp)
            x_test_sp = scaler.transform(x_test_sp)
            clf = SVR(C=params['C'], gamma=params['gamma'], degree=params['degree'], epsilon=params['epsilon'], kernel=params['kernel'])
        elif t == 'lasso':
            clf = Lasso(alpha=params['alpha'])
        elif t == 'ridge':
            clf = Ridge(alpha=params['alpha'])
        clf.fit(x_train_sp, y_train_sp)
        preds = clf.predict(x_test_sp)
        best_score = np.sqrt(mean_squared_error(y_test_sp, preds))
        scores.append(best_score)

    eval_time = datetime.now() - start
    print 'done, {}'.format(eval_time)
    print 'This round\'s loss is {}.'.format(scores)

    return {'loss': np.mean(scores),
            'scores': scores,
            'eval_time': str(eval_time)}

def f(params):
    results = hyperopt_train_test(params.copy())
    return {'loss': results['loss'],
            'status': STATUS_OK,
            'other_stuff': {'scores': results['scores'],
                            'eval_time': results['eval_time']
                            }
            }

space_forest = {
    'n_estimators': 100,
    'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
    'n_jobs': -1,
    'random_state': 22
}
space_svr = hp.choice('rgr_type', [
    {
        'type': 'svr',
        'C': hp.loguniform("C", np.log(1), np.log(100)),
        'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
        'degree': hp.quniform('degree', 1, 5, 1),
        'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),    
        'kernel': hp.choice('kernel', ['rbf', 'poly']),
    }
])

space_lasso = hp.choice('rgr_type', [
    {
        'type': 'lasso',
        'alpha': hp.loguniform("alpha_lasso", np.log(0.00001), np.log(0.1)),
    }
])

space_ridge = hp.choice('rgr_type', [
    {
        'type': 'ridge',
        'alpha': hp.loguniform("alpha_ridge", np.log(0.01), np.log(20)),
    }
])

trials = Trials()
best = fmin(f, space_svr, algo=tpe.suggest, max_evals=1, trials=trials)


###############
# parameters
###############

# random forest
# 'n_estimators': 500
# 'max_features': [0.65]
# 'loss': 0.4582172985246123
# 'eval_time': '0:14:06.325543'
# 'scores': [0.4578581274006101, 0.45604667337328875, 0.46044440766038197, 0.46106480104679692, 0.45567248314198366]

# extra trees
# 'max_features': [0.75]
# 'loss': 0.46877459923688836
# 'eval_time': '0:11:20.235632'
# 'scores': [0.46855086598882006, 0.46722922059472743, 0.46982585307064428, 0.47110515034998351, 0.46716190618026654]

# ridge
# 'alpha_ridge': [0.7907994420477394]
# 'loss': 0.4723708770598546
# 'scores': [0.47087023040770348, 0.4702611961716634, 0.47627576268599781, 0.47509366313731849, 0.46935353289658982]

# lasso
# 'alpha_lasso': [2.6119839093556226e-05]
# 'loss': 0.47236765375485346
# 'scores': [0.47086254079689011, 0.4703244057751208, 0.47627118133562052, 0.47497915358182158, 0.4694009872848145]

