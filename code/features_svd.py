'''

tfidf features using SVD
- transform text to tfidf vec
- transform tfidf vec to SVD feats

features:
1.svd_query
2.svd_title
3.svd_descr
4.svd_brand
5.svd_material
6.svd_color
7.svd_bullet
8.svd_name


'''

import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from datetime import datetime

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

SVD_LENGTH = 10

print '='*20
print 'START EXTRACTING SVD FEATURES...'

print '='*20
print 'loading raw dataframe...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_preprocessed.pkl') as f:
    df_train, df_test = pickle.load(f)

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for query

print '='*20
print 'calculating svd features for query...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['search_term'])
vec_test = tfidf.transform(df_test['search_term'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_query = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_query_{}'.format(x))
df_test_query = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_query_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for product title

print '='*20
print 'calculating svd features for title...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['product_title'], df_test['product_title']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['product_title'])
vec_test = tfidf.transform(df_test['product_title'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_title = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_title_{}'.format(x))
df_test_title = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_title_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for product descr

print '='*20
print 'calculating svd features for description...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['product_description'], df_test['product_description']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['product_description'])
vec_test = tfidf.transform(df_test['product_description'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_descr = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_descr_{}'.format(x))
df_test_descr = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_descr_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for product brand

print '='*20
print 'calculating svd features for brand...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['brand'], df_test['brand']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['brand'])
vec_test = tfidf.transform(df_test['brand'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_brand = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_brand_{}'.format(x))
df_test_brand = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_brand_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for material

print '='*20
print 'calculating svd features for material...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['material'], df_test['material']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['material'])
vec_test = tfidf.transform(df_test['material'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_material = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_material_{}'.format(x))
df_test_material = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_material_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for color

print '='*20
print 'calculating svd features for color...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['color'], df_test['color']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['color'])
vec_test = tfidf.transform(df_test['color'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_color = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_color_{}'.format(x))
df_test_color = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_color_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for bullet

print '='*20
print 'calculating svd features for bullet...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['bullet'], df_test['bullet']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['bullet'])
vec_test = tfidf.transform(df_test['bullet'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_bullet = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_bullet_{}'.format(x))
df_test_bullet = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_bullet_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# SVD for name

print '='*20
print 'calculating svd features for name...'
start = datetime.now()

tfidf = TfidfVectorizer(use_idf=True, norm=None, analyzer=lambda x: x.split())
corpus = pd.concat((df_train['name'], df_test['name']))
tfidf.fit(corpus)

vec_train = tfidf.transform(df_train['name'])
vec_test = tfidf.transform(df_test['name'])

tsvd = TruncatedSVD(n_components=10, random_state=22, n_iter=15)

df_train_name = pd.DataFrame(tsvd.fit_transform(vec_train)).rename(columns=lambda x: 'svd_tfidf_name_{}'.format(x))
df_test_name = pd.DataFrame(tsvd.transform(vec_test)).rename(columns=lambda x: 'svd_tfidf_name_{}'.format(x))

print 'done, {}'.format(datetime.now() - start)

# ======================
# concat dataframes all together

df_train = pd.concat((df_train_query, df_train_title, df_train_descr, df_train_brand, df_train_material, df_train_color, df_train_bullet, df_train_name), axis=1)
df_test = pd.concat((df_test_query, df_test_title, df_test_descr, df_test_brand, df_test_material, df_test_color, df_test_bullet, df_test_name), axis=1)

print '='*20
print 'dumping dataframe to svd pickle...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_features_svd.pkl', 'wb') as f:
    pickle.dump((df_train, df_test), f)

print 'done, {}'.format(datetime.now() - start)

