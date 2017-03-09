'''
features for unigram

- original features are: 
search_term, product_title, product_description, brand, material, color, bullet, name

- pickle stores only new features in dataframe

==========================
features for unigram (unigram/bigram/trigram)
1.len of query/title/descr/brand/material/color/bullet/name (8 feats)
- naming: len_unigram_query

2.hit times in query for title/descr/brand/material/color/bullet/name (7 feats)
- naming: hit_times_unigram_title_in_query

3.hit ratio in query for title/descr/brand/material/color/bullet/name (7 feats)
- naming: hit_ratio_unigram_title_in_query

4.count vec's hit times in title/descr/brand/material/color/bullet/name (7 feats)
- naming: hit_times_count_unigram_title

5.count vec's hit ratio in title/descr/brand/material/color/bullet/name (7 feats)
- naming: hit_ratio_count_unigram_title

6.count vec's cosine distance between query and title/descr/brand/material/color/bullet/name (7 feats)
- naming: cos_count_unigram_title

7.tfidf vec's hit times in title/descr/brand/material/color/bullet/name (7 feats)
- naming: hit_times_tfidf_unigram_title

8.tfidf vec's cosine distance between query and title/descr/brand/material/color/bullet/name (7 feats)
- naming: cos_tfidf_unigram_title

'''

import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import datetime

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

print '='*20
print 'START EXTRACTING UNIGRAM FEATURES...'

print '='*20
print 'loading raw dataframe...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_preprocessed.pkl') as f:
    df_train, df_test = pickle.load(f)

print 'done, {}'.format(datetime.now() - start)

# ======================
# len of unigram for query, title, description, brand
def word_count(text):
    return len(text.strip().split())

print '='*20
print 'calculating length of each text...'
start = datetime.now()

df_train['len_unigram_query'] = df_train['search_term'].apply(word_count)
df_train['len_unigram_title'] = df_train['product_title'].apply(word_count)
df_train['len_unigram_descr'] = df_train['product_description'].apply(word_count)
df_train['len_unigram_brand'] = df_train['brand'].apply(word_count)
df_train['len_unigram_material'] = df_train['material'].apply(word_count)
df_train['len_unigram_color'] = df_train['color'].apply(word_count)
df_train['len_unigram_bullet'] = df_train['bullet'].apply(word_count)
df_train['len_unigram_name'] = df_train['name'].apply(word_count)


df_test['len_unigram_query'] = df_test['search_term'].apply(word_count)
df_test['len_unigram_title'] = df_test['product_title'].apply(word_count)
df_test['len_unigram_descr'] = df_test['product_description'].apply(word_count)
df_test['len_unigram_brand'] = df_test['brand'].apply(word_count)
df_test['len_unigram_material'] = df_test['material'].apply(word_count)
df_test['len_unigram_color'] = df_test['color'].apply(word_count)
df_test['len_unigram_bullet'] = df_test['bullet'].apply(word_count)
df_test['len_unigram_name'] = df_test['name'].apply(word_count)


print 'done, {}'.format(datetime.now() - start)

# ======================
# hit times (= 0/1_vec_query * 0/1_vec_title/desc/brand)
vectorizer = TfidfVectorizer(binary=True, use_idf=False, norm=None)
corpus = pd.concat((df_train['search_term'], df_test['search_term'])) # only consider vocabulary of queries, and stop words are not considered.
vectorizer.fit(corpus)

print '='*20
print 'calculating 0/1 vec of each text...'
start = datetime.now()

df_train['01_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['01_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['01_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['01_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['01_vec_material'] = [row for row in vectorizer.transform(df_train['material'])]
df_train['01_vec_color'] = [row for row in vectorizer.transform(df_train['color'])]
df_train['01_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['01_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['01_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['01_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['01_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['01_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['01_vec_material'] = [row for row in vectorizer.transform(df_test['material'])]
df_test['01_vec_color'] = [row for row in vectorizer.transform(df_test['color'])]
df_test['01_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['01_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]


print 'done, {}'.format(datetime.now() - start)

print '='*20
print 'calculating hit times in query for title/descr/brand...'
start = datetime.now()

df_train['hit_times_unigram_title_in_query'] = df_train[['01_vec_query', '01_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_descr_in_query'] = df_train[['01_vec_query', '01_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_brand_in_query'] = df_train[['01_vec_query', '01_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_material_in_query'] = df_train[['01_vec_query', '01_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_color_in_query'] = df_train[['01_vec_query', '01_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_bullet_in_query'] = df_train[['01_vec_query', '01_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_unigram_name_in_query'] = df_train[['01_vec_query', '01_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


df_test['hit_times_unigram_title_in_query'] = df_test[['01_vec_query', '01_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_descr_in_query'] = df_test[['01_vec_query', '01_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_brand_in_query'] = df_test[['01_vec_query', '01_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_material_in_query'] = df_test[['01_vec_query', '01_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_color_in_query'] = df_test[['01_vec_query', '01_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_bullet_in_query'] = df_test[['01_vec_query', '01_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_unigram_name_in_query'] = df_test[['01_vec_query', '01_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


print 'done, {}'.format(datetime.now() - start)

# ======================
# hit ratio (= hit_times / len_query)

def hit_ratio(hit_times, len_text):
    if not len_text:
        return 0 # if len_query is 0 return 0
    else:
        return hit_times/len_text # if len_query is not 0 return the score

print '='*20
print 'calculating hit ratio in query for title/descr/brand...'
start = datetime.now()

df_train['hit_ratio_unigram_title_in_query'] = df_train[['hit_times_unigram_title_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_unigram_descr_in_query'] = df_train[['hit_times_unigram_descr_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_unigram_brand_in_query'] = df_train[['hit_times_unigram_brand_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_unigram_material_in_query'] = df_train[['hit_times_unigram_material_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_unigram_color_in_query'] = df_train[['hit_times_unigram_color_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1) 
df_train['hit_ratio_unigram_bullet_in_query'] = df_train[['hit_times_unigram_bullet_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_unigram_name_in_query'] = df_train[['hit_times_unigram_name_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)


df_test['hit_ratio_unigram_title_in_query'] = df_test[['hit_times_unigram_title_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_descr_in_query'] = df_test[['hit_times_unigram_descr_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_brand_in_query'] = df_test[['hit_times_unigram_brand_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_material_in_query'] = df_test[['hit_times_unigram_material_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_color_in_query'] = df_test[['hit_times_unigram_color_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_bullet_in_query'] = df_test[['hit_times_unigram_bullet_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_unigram_name_in_query'] = df_test[['hit_times_unigram_name_in_query', 'len_unigram_query']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)


print 'done, {}'.format(datetime.now() - start)

# # ======================
# # jaccard distance
# # depreciated: given the new vocabulary, jaccard is the same as hit ratio
# from scipy.spatial.distance import jaccard

# print '='*20
# print 'calculating jaccard distance...'
# start = datetime.now()

# df_train['jaccard_unigram_title'] = df_train[['01_vec_query', '01_vec_title']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_train['jaccard_unigram_descr'] = df_train[['01_vec_query', '01_vec_descr']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_train['jaccard_unigram_brand'] = df_train[['01_vec_query', '01_vec_brand']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

# df_test['jaccard_unigram_title'] = df_test[['01_vec_query', '01_vec_title']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_test['jaccard_unigram_descr'] = df_test[['01_vec_query', '01_vec_descr']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
# df_test['jaccard_unigram_brand'] = df_test[['01_vec_query', '01_vec_brand']].apply(lambda x: jaccard(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)

# print 'done, {}'.format(datetime.now() - start)


# ======================
# count vec
vectorizer = TfidfVectorizer(use_idf=False, norm=None)
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
vectorizer.fit(corpus)

print '='*20
print 'calculating count vec of each text...'
start = datetime.now()

df_train['count_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['count_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['count_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['count_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['count_vec_material'] = [row for row in vectorizer.transform(df_train['material'])]
df_train['count_vec_color'] = [row for row in vectorizer.transform(df_train['color'])]
df_train['count_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['count_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['count_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['count_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['count_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['count_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['count_vec_material'] = [row for row in vectorizer.transform(df_test['material'])]
df_test['count_vec_color'] = [row for row in vectorizer.transform(df_test['color'])]
df_test['count_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['count_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]


print 'done, {}'.format(datetime.now() - start)

# ======================
# hit times in title/descr/brand

print '='*20
print 'calculating count vec hit times in title/descr/brand...'
start = datetime.now()

df_train['hit_times_count_unigram_title'] = df_train[['01_vec_query', 'count_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_descr'] = df_train[['01_vec_query', 'count_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_brand'] = df_train[['01_vec_query', 'count_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_material'] = df_train[['01_vec_query', 'count_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_color'] = df_train[['01_vec_query', 'count_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_bullet'] = df_train[['01_vec_query', 'count_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_count_unigram_name'] = df_train[['01_vec_query', 'count_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


df_test['hit_times_count_unigram_title'] = df_test[['01_vec_query', 'count_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_descr'] = df_test[['01_vec_query', 'count_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_brand'] = df_test[['01_vec_query', 'count_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_material'] = df_test[['01_vec_query', 'count_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_color'] = df_test[['01_vec_query', 'count_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_bullet'] = df_test[['01_vec_query', 'count_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_count_unigram_name'] = df_test[['01_vec_query', 'count_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


print 'done, {}'.format(datetime.now() - start)

# ======================
# hit ratio in title/descr/brand
def hit_ratio(hit_times, len_text):
    if not len_text:
        return 0 # if len_query is 0 return 0
    else:
        return hit_times/len_text # if len_query is not 0 return the score

print '='*20
print 'calculating count vec hit ratio in title/descr/brand...'
start = datetime.now()

df_train['hit_ratio_count_unigram_title'] = df_train[['hit_times_count_unigram_title', 'len_unigram_title']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_descr'] = df_train[['hit_times_count_unigram_descr', 'len_unigram_descr']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_brand'] = df_train[['hit_times_count_unigram_brand', 'len_unigram_brand']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_material'] = df_train[['hit_times_count_unigram_material', 'len_unigram_material']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_color'] = df_train[['hit_times_count_unigram_color', 'len_unigram_color']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_bullet'] = df_train[['hit_times_count_unigram_bullet', 'len_unigram_bullet']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_train['hit_ratio_count_unigram_name'] = df_train[['hit_times_count_unigram_name', 'len_unigram_name']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)


df_test['hit_ratio_count_unigram_title'] = df_test[['hit_times_count_unigram_title', 'len_unigram_title']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_descr'] = df_test[['hit_times_count_unigram_descr', 'len_unigram_descr']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_brand'] = df_test[['hit_times_count_unigram_brand', 'len_unigram_brand']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_material'] = df_test[['hit_times_count_unigram_material', 'len_unigram_material']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_color'] = df_test[['hit_times_count_unigram_color', 'len_unigram_color']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_bullet'] = df_test[['hit_times_count_unigram_bullet', 'len_unigram_bullet']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)
df_test['hit_ratio_count_unigram_name'] = df_test[['hit_times_count_unigram_name', 'len_unigram_name']].apply(lambda x: hit_ratio(x[0], x[1]), axis=1)

print 'done, {}'.format(datetime.now() - start)

# ======================
# count vec's cosine distance
from scipy.spatial.distance import cosine

print '='*20
print 'calculating cosine distance of count vec...'
start = datetime.now()

df_train['cos_count_unigram_title'] = df_train[['01_vec_query', 'count_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_descr'] = df_train[['01_vec_query', 'count_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_brand'] = df_train[['01_vec_query', 'count_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_material'] = df_train[['01_vec_query', 'count_vec_material']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_color'] = df_train[['01_vec_query', 'count_vec_color']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_bullet'] = df_train[['01_vec_query', 'count_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_count_unigram_name'] = df_train[['01_vec_query', 'count_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)


df_test['cos_count_unigram_title'] = df_test[['01_vec_query', 'count_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_descr'] = df_test[['01_vec_query', 'count_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_brand'] = df_test[['01_vec_query', 'count_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_material'] = df_test[['01_vec_query', 'count_vec_material']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_color'] = df_test[['01_vec_query', 'count_vec_color']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_bullet'] = df_test[['01_vec_query', 'count_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_count_unigram_name'] = df_test[['01_vec_query', 'count_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)


print 'done, {}'.format(datetime.now() - start)

# ======================
# tfidf vec
vectorizer = TfidfVectorizer(norm=None)
corpus = pd.concat((df_train['search_term'], df_test['search_term']))
vectorizer.fit(corpus)

print '='*20
print 'calculating tfidf vec of each text...'
start = datetime.now()

df_train['tfidf_vec_query'] = [row for row in vectorizer.transform(df_train['search_term'])]
df_train['tfidf_vec_title'] = [row for row in vectorizer.transform(df_train['product_title'])]
df_train['tfidf_vec_descr'] = [row for row in vectorizer.transform(df_train['product_description'])]
df_train['tfidf_vec_brand'] = [row for row in vectorizer.transform(df_train['brand'])]
df_train['tfidf_vec_material'] = [row for row in vectorizer.transform(df_train['material'])]
df_train['tfidf_vec_color'] = [row for row in vectorizer.transform(df_train['color'])]
df_train['tfidf_vec_bullet'] = [row for row in vectorizer.transform(df_train['bullet'])]
df_train['tfidf_vec_name'] = [row for row in vectorizer.transform(df_train['name'])]


df_test['tfidf_vec_query'] = [row for row in vectorizer.transform(df_test['search_term'])]
df_test['tfidf_vec_title'] = [row for row in vectorizer.transform(df_test['product_title'])]
df_test['tfidf_vec_descr'] = [row for row in vectorizer.transform(df_test['product_description'])]
df_test['tfidf_vec_brand'] = [row for row in vectorizer.transform(df_test['brand'])]
df_test['tfidf_vec_material'] = [row for row in vectorizer.transform(df_test['material'])]
df_test['tfidf_vec_color'] = [row for row in vectorizer.transform(df_test['color'])]
df_test['tfidf_vec_bullet'] = [row for row in vectorizer.transform(df_test['bullet'])]
df_test['tfidf_vec_name'] = [row for row in vectorizer.transform(df_test['name'])]


print 'done, {}'.format(datetime.now() - start)

# ======================
# tfidf hit times

print '='*20
print 'calculating tfidf vec hit times in title/descr/brand...'
start = datetime.now()

df_train['hit_times_tfidf_unigram_title'] = df_train[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_descr'] = df_train[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_brand'] = df_train[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_material'] = df_train[['01_vec_query', 'tfidf_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_bullet'] = df_train[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_color'] = df_train[['01_vec_query', 'tfidf_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_train['hit_times_tfidf_unigram_name'] = df_train[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


df_test['hit_times_tfidf_unigram_title'] = df_test[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_descr'] = df_test[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_brand'] = df_test[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_material'] = df_test[['01_vec_query', 'tfidf_vec_material']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_bullet'] = df_test[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_color'] = df_test[['01_vec_query', 'tfidf_vec_color']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)
df_test['hit_times_tfidf_unigram_name'] = df_test[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: x[0].dot(x[1].transpose(True)).toarray()[0, 0], axis=1)


print 'done, {}'.format(datetime.now() - start)

# ======================
# tfidf cosine distance
print '='*20
print 'calculating cosine distance of tfidf vec...'
start = datetime.now()

df_train['cos_tfidf_unigram_title'] = df_train[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_descr'] = df_train[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_brand'] = df_train[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_material'] = df_train[['01_vec_query', 'tfidf_vec_material']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_color'] = df_train[['01_vec_query', 'tfidf_vec_color']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_bullet'] = df_train[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_train['cos_tfidf_unigram_name'] = df_train[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)


df_test['cos_tfidf_unigram_title'] = df_test[['01_vec_query', 'tfidf_vec_title']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_descr'] = df_test[['01_vec_query', 'tfidf_vec_descr']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_brand'] = df_test[['01_vec_query', 'tfidf_vec_brand']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_material'] = df_test[['01_vec_query', 'tfidf_vec_material']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_color'] = df_test[['01_vec_query', 'tfidf_vec_color']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_bullet'] = df_test[['01_vec_query', 'tfidf_vec_bullet']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)
df_test['cos_tfidf_unigram_name'] = df_test[['01_vec_query', 'tfidf_vec_name']].apply(lambda x: cosine(x[0].toarray().flatten(), x[1].toarray().flatten()), axis=1).fillna(1.0)


print 'done, {}'.format(datetime.now() - start)

# original columns need to be dropped
# only keep the unigram relevant features
col_names_train = [u'id', u'product_uid', u'product_title', u'search_term', u'relevance', u'product_description', u'brand', u'material', u'color', u'bullet', u'name', u'01_vec_query', u'01_vec_title', u'01_vec_descr', u'01_vec_brand', u'01_vec_material', u'01_vec_color', u'01_vec_bullet', u'01_vec_name', u'count_vec_query', u'count_vec_title', u'count_vec_descr', u'count_vec_brand', u'count_vec_material', u'count_vec_color', u'count_vec_bullet', u'count_vec_name', u'tfidf_vec_query', u'tfidf_vec_title', u'tfidf_vec_descr', u'tfidf_vec_brand', u'tfidf_vec_material', u'tfidf_vec_color', u'tfidf_vec_bullet', u'tfidf_vec_name']

col_names_test = [u'id', u'product_uid', u'product_title', u'search_term', u'product_description', u'brand', u'material', u'color', u'bullet', u'name', u'01_vec_query', u'01_vec_title', u'01_vec_descr', u'01_vec_brand', u'01_vec_material', u'01_vec_color', u'01_vec_bullet', u'01_vec_name', u'count_vec_query', u'count_vec_title', u'count_vec_descr', u'count_vec_brand', u'count_vec_material', u'count_vec_color', u'count_vec_bullet', u'count_vec_name', u'tfidf_vec_query', u'tfidf_vec_title', u'tfidf_vec_descr', u'tfidf_vec_brand', u'tfidf_vec_material', u'tfidf_vec_color', u'tfidf_vec_bullet', u'tfidf_vec_name']

df_train.drop(labels=col_names_train, axis=1, inplace=True)
df_test.drop(labels=col_names_test, axis=1, inplace=True)

print '='*20
print 'training data has features: {}'.format('\t'.join(df_train.columns))
print '='*20
print 'test data has features: {}'.format('\t'.join(df_test.columns))

print '='*20
print 'dumping dataframe to pickle...'
start = datetime.now()

with open(PROJECT_PATH + 'pickles/df_features_unigram.pkl', 'wb') as f:
    pickle.dump((df_train, df_test), f)

print 'done, {}'.format(datetime.now() - start)
