'''
features for trigram

'''

import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import datetime

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

print '='*20
print 'START EXTRACTING TRIGRAM FEATURES...'

print '='*20
print 'loading raw dataframe...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_preprocessed.pkl') as f:
    df_train, df_test = pickle.load(f)

print 'done, {}'.format(datetime.now() - start)

# ======================
# text to 0/1 vec, using full vocabulary to get len of each text
vectorizer = TfidfVectorizer(binary=True, use_idf=False, norm=None, ngram_range=(3, 3))
corpus = pd.concat((df_train['search_term'], df_test['search_term'], df_train['product_title'], df_test['product_title'], df_train['product_description'], df_test['product_description'], df_train['brand'], df_test['brand'], df_train['bullet'], df_test['bullet'], df_train['name'], df_test['name']))
vectorizer.fit(corpus)

print '='*20
print 'calculating 0/1 vec of each text (full vocabulary)...'
start = datetime.now()

df_train['01_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['01_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['01_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['01_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['01_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['01_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['01_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['01_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['01_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['01_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['01_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['01_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]

print 'done, {}'.format(datetime.now() - start)

# ======================
# len of trigram for query, title, description, brand

print '='*20
print 'calculating length of each trigram text...'
start = datetime.now()

df_train['len_trigram_query'] = df_train['01_vec_query'].apply(lambda x: x.sum())
df_train['len_trigram_title'] = df_train['01_vec_title'].apply(lambda x: x.sum())
df_train['len_trigram_descr'] = df_train['01_vec_descr'].apply(lambda x: x.sum())
df_train['len_trigram_brand'] = df_train['01_vec_brand'].apply(lambda x: x.sum())
df_train['len_trigram_bullet'] = df_train['01_vec_bullet'].apply(lambda x: x.sum())
df_train['len_trigram_name'] = df_train['01_vec_name'].apply(lambda x: x.sum())

df_test['len_trigram_query'] = df_test['01_vec_query'].apply(lambda x: x.sum())
df_test['len_trigram_title'] = df_test['01_vec_title'].apply(lambda x: x.sum())
df_test['len_trigram_descr'] = df_test['01_vec_descr'].apply(lambda x: x.sum())
df_test['len_trigram_brand'] = df_test['01_vec_brand'].apply(lambda x: x.sum())
df_test['len_trigram_bullet'] = df_test['01_vec_bullet'].apply(lambda x: x.sum())
df_test['len_trigram_name'] = df_test['01_vec_name'].apply(lambda x: x.sum())

print 'done, {}'.format(datetime.now() - start)

# ======================
# text to 0/1 vec, using only query's vocabulary to simplify computation
vectorizer = TfidfVectorizer(binary=True, use_idf=False, norm=None, ngram_range=(3, 3))
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
vectorizer.fit(corpus)

print '='*20
print 'calculating 0/1 vec of each text...'
start = datetime.now()

df_train['01_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['01_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['01_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['01_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['01_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['01_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['01_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['01_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['01_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['01_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['01_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['01_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]

print 'done, {}'.format(datetime.now() - start)


# ======================
# hit times (= 0/1_vec_query * 0/1_vec_title/desc/brand)

print '='*20
print 'calculating hit times...'
start = datetime.now()

df_train['hit_times_trigram_title_in_query'] = df_train[['01_vec_query', '01_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_train['hit_times_trigram_descr_in_query'] = df_train[['01_vec_query', '01_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_train['hit_times_trigram_brand_in_query'] = df_train[['01_vec_query', '01_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_train['hit_times_trigram_bullet_in_query'] = df_train[['01_vec_query', '01_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_train['hit_times_trigram_name_in_query'] = df_train[['01_vec_query', '01_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)


df_test['hit_times_trigram_title_in_query'] = df_test[['01_vec_query', '01_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_test['hit_times_trigram_descr_in_query'] = df_test[['01_vec_query', '01_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_test['hit_times_trigram_brand_in_query'] = df_test[['01_vec_query', '01_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_test['hit_times_trigram_bullet_in_query'] = df_test[['01_vec_query', '01_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)
df_test['hit_times_trigram_name_in_query'] = df_test[['01_vec_query', '01_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(copy=True)).toarray()[0, 0], axis=1)


print 'done, {}'.format(datetime.now() - start)

# ======================
# hit ratio (= hit_times / len_query if len_query is nonzero else 0)

def hit_ratio(hit_times, len_text):
    if not len_text:
        return 0 # if len_query is 0 return 0
    else:
        return hit_times/len_text # if len_query is not 0 return the score

print '='*20
print 'calculating trigram hit ratio...'
start = datetime.now()

df_train['hit_ratio_trigram_title_in_query'] = df_train[['hit_times_trigram_title_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_trigram_descr_in_query'] = df_train[['hit_times_trigram_descr_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_trigram_brand_in_query'] = df_train[['hit_times_trigram_brand_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_trigram_bullet_in_query'] = df_train[['hit_times_trigram_bullet_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_trigram_name_in_query'] = df_train[['hit_times_trigram_name_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)

df_test['hit_ratio_trigram_title_in_query'] = df_test[['hit_times_trigram_title_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_trigram_descr_in_query'] = df_test[['hit_times_trigram_descr_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_trigram_brand_in_query'] = df_test[['hit_times_trigram_brand_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_trigram_bullet_in_query'] = df_test[['hit_times_trigram_bullet_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_trigram_name_in_query'] = df_test[['hit_times_trigram_name_in_query', 'len_trigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)

print 'done, {}'.format(datetime.now() - start)

# # ======================
# # jaccard distance
# # depreciated: given the new vocabulary, jaccard is the same as hit ratio
# from scipy.spatial.distance import jaccard

# print '='*20
# print 'calculating bigram jaccard distance...'
# start = datetime.now()

# df_train['jaccard_bigram_title'] = df_train[['vec_query', 'vec_title']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_train['jaccard_bigram_descr'] = df_train[['vec_query', 'vec_descr']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_train['jaccard_bigram_brand'] = df_train[['vec_query', 'vec_brand']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

# df_test['jaccard_bigram_title'] = df_test[['vec_query', 'vec_title']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_test['jaccard_bigram_descr'] = df_test[['vec_query', 'vec_descr']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_test['jaccard_bigram_brand'] = df_test[['vec_query', 'vec_brand']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

# print 'done, {}'.format(datetime.now() - start)

# ======================
# trigram count vec
vectorizer = TfidfVectorizer(use_idf=False, norm=None, ngram_range=(3, 3))
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
vectorizer.fit(corpus)

print '='*20
print 'calculating trigram count vec of each text...'
start = datetime.now()

df_train['count_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['count_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['count_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['count_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['count_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['count_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]

df_test['count_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['count_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['count_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['count_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['count_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['count_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]

print 'done, {}'.format(datetime.now() - start)

# ======================
# hit times in title/descr/brand

print '='*20
print 'calculating count vec hit times in title/descr/brand...'
start = datetime.now()

df_train['hit_times_count_trigram_title'] = df_train[['01_vec_query', 'count_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_trigram_descr'] = df_train[['01_vec_query', 'count_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_trigram_brand'] = df_train[['01_vec_query', 'count_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_trigram_bullet'] = df_train[['01_vec_query', 'count_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_trigram_name'] = df_train[['01_vec_query', 'count_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)

df_test['hit_times_count_trigram_title'] = df_test[['01_vec_query', 'count_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_trigram_descr'] = df_test[['01_vec_query', 'count_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_trigram_brand'] = df_test[['01_vec_query', 'count_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_trigram_bullet'] = df_test[['01_vec_query', 'count_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_trigram_name'] = df_test[['01_vec_query', 'count_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)

print 'done, {}'.format(datetime.now() - start)

# ======================
# hit ratio in title/descr/brand

print '='*20
print 'calculating count vec hit ratio in title/descr/brand...'
start = datetime.now()

df_train['hit_ratio_count_trigram_title'] = df_train[['hit_times_count_trigram_title', 'len_trigram_title']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_trigram_descr'] = df_train[['hit_times_count_trigram_descr', 'len_trigram_descr']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_trigram_brand'] = df_train[['hit_times_count_trigram_brand', 'len_trigram_brand']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_trigram_bullet'] = df_train[['hit_times_count_trigram_bullet', 'len_trigram_bullet']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_trigram_name'] = df_train[['hit_times_count_trigram_name', 'len_trigram_name']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)

df_test['hit_ratio_count_trigram_title'] = df_test[['hit_times_count_trigram_title', 'len_trigram_title']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_trigram_descr'] = df_test[['hit_times_count_trigram_descr', 'len_trigram_descr']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_trigram_brand'] = df_test[['hit_times_count_trigram_brand', 'len_trigram_brand']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_trigram_bullet'] = df_test[['hit_times_count_trigram_bullet', 'len_trigram_bullet']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_trigram_name'] = df_test[['hit_times_count_trigram_name', 'len_trigram_name']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)

print 'done, {}'.format(datetime.now() - start)

# ======================
# count vec's cosine distance
from scipy.spatial.distance import cosine

print '='*20
print 'calculating cosine distance of count vec...'
start = datetime.now()

df_train['cos_count_trigram_title'] = df_train[['01_vec_query', 'count_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_trigram_descr'] = df_train[['01_vec_query', 'count_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_trigram_brand'] = df_train[['01_vec_query', 'count_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_trigram_bullet'] = df_train[['01_vec_query', 'count_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_trigram_name'] = df_train[['01_vec_query', 'count_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

df_test['cos_count_trigram_title'] = df_test[['01_vec_query', 'count_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_trigram_descr'] = df_test[['01_vec_query', 'count_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_trigram_brand'] = df_test[['01_vec_query', 'count_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_trigram_bullet'] = df_test[['01_vec_query', 'count_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_trigram_name'] = df_test[['01_vec_query', 'count_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)


print 'done, {}'.format(datetime.now() - start)


# ======================
# tfidf vec
vectorizer = TfidfVectorizer(norm=None, ngram_range=(3, 3))
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
vectorizer.fit(corpus)

print '='*20
print 'calculating tfidf vec of each text...'
start = datetime.now()

df_train['tfidf_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['tfidf_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['tfidf_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['tfidf_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['tfidf_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['tfidf_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['tfidf_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['tfidf_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['tfidf_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['tfidf_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['tfidf_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['tfidf_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]

print 'done, {}'.format(datetime.now() - start)

# ======================
# tfidf hit times

print '='*20
print 'calculating tfidf vec hit times in title/descr/brand...'
start = datetime.now()

df_train['hit_times_tfidf_trigram_title'] = df_train[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_trigram_descr'] = df_train[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_trigram_brand'] = df_train[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_trigram_bullet'] = df_train[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_trigram_name'] = df_train[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)

df_test['hit_times_tfidf_trigram_title'] = df_test[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_trigram_descr'] = df_test[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_trigram_brand'] = df_test[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_trigram_bullet'] = df_test[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_trigram_name'] = df_test[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)

print 'done, {}'.format(datetime.now() - start)

# ======================
# tfidf cosine distance
print '='*20
print 'calculating cosine distance of tfidf vec...'
start = datetime.now()

df_train['cos_tfidf_trigram_title'] = df_train[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_trigram_descr'] = df_train[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_trigram_brand'] = df_train[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_trigram_bullet'] = df_train[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_trigram_name'] = df_train[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

df_test['cos_tfidf_trigram_title'] = df_test[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_trigram_descr'] = df_test[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_trigram_brand'] = df_test[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_trigram_bullet'] = df_test[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_trigram_name'] = df_test[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

print 'done, {}'.format(datetime.now() - start)

# original columns need to be dropped
# only keep the trigram relevant features
col_names_train = [u'id', u'product_uid', u'product_title', u'search_term', u'relevance', u'product_description', u'brand', u'material', u'color', u'bullet', u'name', u'01_vec_query', u'01_vec_title', u'01_vec_descr', u'01_vec_brand', u'01_vec_bullet', u'01_vec_name', u'count_vec_query', u'count_vec_title', u'count_vec_descr', u'count_vec_brand', u'count_vec_bullet', u'count_vec_name', u'tfidf_vec_query', u'tfidf_vec_title', u'tfidf_vec_descr', u'tfidf_vec_brand', u'tfidf_vec_bullet', u'tfidf_vec_name']
col_names_test = [u'id', u'product_uid', u'product_title', u'search_term', u'product_description', u'brand', u'material', u'color', u'bullet', u'name', u'01_vec_query', u'01_vec_title', u'01_vec_descr', u'01_vec_brand', u'01_vec_bullet', u'01_vec_name', u'count_vec_query', u'count_vec_title', u'count_vec_descr', u'count_vec_brand', u'count_vec_bullet', u'count_vec_name', u'tfidf_vec_query', u'tfidf_vec_title', u'tfidf_vec_descr', u'tfidf_vec_brand', u'tfidf_vec_bullet', u'tfidf_vec_name']

df_train.drop(labels=col_names_train, axis=1, inplace=True)
df_test.drop(labels=col_names_test, axis=1, inplace=True)

print '='*20
print 'training data has features: {}'.format('\t'.join(df_train.columns))
print '='*20
print 'test data has features: {}'.format('\t'.join(df_test.columns))

print '='*20
print 'dumping dataframe to trigram pickle...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_features_trigram.pkl', 'wb') as f:
    pickle.dump((df_train, df_test), f)

print 'done, {}'.format(datetime.now() - start)



