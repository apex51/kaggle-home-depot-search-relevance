import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils



from datetime import datetime

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

PROJECT_PATH = '/home/jiangh/home_depot/'

# #############################
# # loading data
# #############################

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

def hyperopt_train_test(param):
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

        model = Sequential()
        # first layer
        model.add(Dense(param["hidden_units_first"], input_dim=x_train.shape[1], init='glorot_uniform', activation=param['activation']))
        model.add(Dropout(param["hidden_dropout_first"]))
        # second layer
        model.add(Dense(param["hidden_units_second"], init='glorot_uniform', activation=param['activation']))
        model.add(Dropout(param["hidden_dropout_second"]))
        # output layer
        model.add(Dense(1, init='glorot_uniform', activation='linear'))

        model.compile(loss='mean_squared_error', optimizer="adam")

        scaler = StandardScaler()
        x_train_sp = scaler.fit_transform(x_train_sp)
        x_test_sp = scaler.transform(x_test_sp)

        model.fit(x_train_sp, y_train_sp, nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], validation_split=0)

        preds = model.predict(x_test_sp)

        best_score = np.sqrt(mean_squared_error(y_test_sp, preds))
        scores.append(best_score)

    eval_time = datetime.now() - start
    print 'done, {}'.format(eval_time)
    print 'This round\'s loss is {}.'.format(scores)

    return {'loss': np.mean(scores), 'scores': scores, 'eval_time': str(eval_time)}

def f(params):
    results = hyperopt_train_test(params.copy())
    return {'loss': results['loss'], 'status': STATUS_OK, 'other_stuff': {'scores': results['scores'], 'eval_time': results['eval_time']} }

space = {
    'hidden_units_first': hp.choice('hidden_units_first', [128]),
    'activation': hp.choice('activation', ['sigmoid']),
    'hidden_dropout_first': hp.choice('hidden_dropout_first', [0.5]),
    'hidden_units_second': hp.choice('hidden_units_second', [256]),
    'hidden_dropout_second': hp.quniform('hidden_dropout_second', 0, 0.9, 0.1),
    'batch_size': hp.choice('batch_size', [64]),
    'nb_epoch': hp.choice('nb_epoch', [20, 30, 40, 50, 60]),
}

# space = {
#     'hidden_units_first': hp.choice('hidden_units_first', [64, 128, 256, 512]),
#     'activation': hp.choice('activation', ['sigmoid']),
#     'hidden_dropout_first': hp.quniform('hidden_dropout_first', 0, 0.9, 0.1),
#     'hidden_units_second': hp.choice('hidden_units_second', [64, 128, 256, 512]),
#     'hidden_dropout_second': hp.quniform('hidden_dropout_second', 0, 0.9, 0.1),
#     'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
#     'nb_epoch': hp.choice('nb_epoch', [10, 20, 30, 40, 50]),
# }


trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=15, trials=trials)

#################
# use random forest
# model = Sequential()

# # first layer
# model.add(Dense(128, input_dim=x_train.shape[1], init='glorot_uniform', activation='relu'))
# model.add(Dropout(0.5))
# # second layer
# model.add(Dense(256, init='glorot_uniform', activation='relu'))
# model.add(Dropout(0.0))
# # output layer
# model.add(Dense(1, init='glorot_uniform', activation='linear'))

# model.compile(loss='mean_squared_error', optimizer="adam")

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# model.fit(x_train, y_train, nb_epoch=30, batch_size=64, validation_split=0, verbose=0)
# preds = model.predict(x_test, verbose=0)

# df_preds = pd.read_csv(PROJECT_PATH + 'data/sample_submission.csv')
# df_preds['relevance'] = preds
# df_preds.loc[df_preds['relevance'] < 1, 'relevance'] = 1
# df_preds.loc[df_preds['relevance'] > 3, 'relevance'] = 3
# df_preds.to_csv(PROJECT_PATH + 'result/submission.csv', index=False)


########################## one example
#'activation': 'relu',
#'batch_size': 64,
#'hidden_dropout_first': 0.5,
#'hidden_dropout_second': 0.0,
#'hidden_units_first': 128,
#'hidden_units_second': 256,
#'nb_epoch': 30,

#'loss': 0.46385482040917275,
#'eval_time': '0:05:39.990399',
#'scores': [0.46391625722609214, 0.46141193997450236, 0.4671414810569941, 0.46752344163111392, 0.45928098215716134],

########################## another one
# 'activation': 'sigmoid',
# 'batch_size': 64,
# 'hidden_dropout_first': 0.5,
# 'hidden_dropout_second': 0.2,
# 'hidden_units_first': 128,
# 'hidden_units_second': 256,
# 'nb_epoch': 50,

# 'loss': 0.4635309773397943,
# 'eval_time': '0:09:57.554997',
# 'scores': [0.46406551539774749, 0.45816948231810201, 0.46639349945290004, 0.46743987848555163, 0.46158651104467063]},
