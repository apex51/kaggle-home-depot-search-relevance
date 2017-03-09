import numpy as np
import pandas as pd
import pickle
import re

from datetime import datetime
print '='*20
print 'feature extracting...'
start = datetime.now()

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

with open(PROJECT_PATH + 'pickles/df_preprocessed.pkl') as f:
    df_train, df_test = pickle.load(f)

# convenient func to find how many words (in title/description) hit the query
def words_counter(strs):
    str_target, str_words = strs
    return sum(int(str_words.find(word) >= 0) for word in str_target.split())

# feature #1: count words in title which hit the query
df_train['nwords_in_title'] = df_train[['product_title', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_title'] = df_test[['product_title', 'search_term']].apply(words_counter, axis=1)
# feature #2: count words in description which hit the query
df_train['nwords_in_description'] = df_train[['product_description', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_description'] = df_test[['product_description', 'search_term']].apply(words_counter, axis=1)

# feature #3: cout search words
df_train['nwords_search'] = df_train['search_term'].apply(lambda x: len(x.split()))
df_test['nwords_search'] = df_test['search_term'].apply(lambda x: len(x.split()))

from sklearn.feature_extraction.text import TfidfVectorizer

# covenient func to count sum of tfidf values
def tfidf_counter(strs, vectorizer):
    tfidf_vec, str_words = strs
    tfidf_vec = tfidf_vec.A.flatten()
    return sum(tfidf_vec[vectorizer.vocabulary_.get(word)] for word in str_words.split() if vectorizer.vocabulary_.get(word))

# feature #4: count using tfidf in title
# title_tfidf_word_vector * search_term_word_vector
vectorizer = TfidfVectorizer()
vectorizer.fit(pd.concat((df_train['product_title'], df_test['product_title'])))
df_train['tfidf_title'] = [row for row in vectorizer.transform(df_train['product_title'])] # sparse
df_train['nwords_tfidf_title'] = df_train[['tfidf_title', 'search_term']].apply(lambda x: tfidf_counter(x, vectorizer), axis=1)
df_test['tfidf_title'] = [row for row in vectorizer.transform(df_test['product_title'])] # sparse
df_test['nwords_tfidf_title'] = df_test[['tfidf_title', 'search_term']].apply(lambda x: tfidf_counter(x, vectorizer), axis=1)

# feature #5: count using tfidf in description
vectorizer = TfidfVectorizer()
vectorizer.fit(pd.concat((df_train['product_description'], df_test['product_description'])))
df_train['tfidf_description'] = [row for row in vectorizer.transform(df_train['product_description'])] # sparse
df_train['nwords_tfidf_description'] = df_train[['tfidf_description', 'search_term']].apply(lambda x: tfidf_counter(x, vectorizer), axis=1)
df_test['tfidf_description'] = [row for row in vectorizer.transform(df_test['product_description'])] # sparse
df_test['nwords_tfidf_description'] = df_test[['tfidf_description', 'search_term']].apply(lambda x: tfidf_counter(x, vectorizer), axis=1)

# feature #6: count title
df_train['nwords_title'] = df_train['product_title'].apply(lambda x: len(x.split()))
df_test['nwords_title'] = df_test['product_title'].apply(lambda x: len(x.split()))

# feature #7: count description
df_train['nwords_description'] = df_train['product_description'].apply(lambda x: len(x.split()))
df_test['nwords_description'] = df_test['product_description'].apply(lambda x: len(x.split()))

# convenient func to find how many words in the query hit the title/description
# for feature #8 and #9
def words_counter_in_query(strs):
    str_target, str_words = strs
    return sum(int(str_target.find(word) >= 0) for word in str_words.split())

# feature #8: count words in query which hit the title
df_train['nwords_in_query_hit_title'] = df_train[['product_title', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_title'] = df_test[['product_title', 'search_term']].apply(words_counter_in_query, axis=1)

# feature #9: count words in query which hit the description
df_train['nwords_in_query_hit_description'] = df_train[['product_description', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_description'] = df_test[['product_description', 'search_term']].apply(words_counter_in_query, axis=1)

# convenient func to find how many times query appears in title/description
def query_counter(strs):
    str_target, str_words = strs
    return str_target.count(str_words)

# feature #10: count how many times query appeared in title
df_train['nquerys_in_title'] = df_train[['product_title', 'search_term']].apply(query_counter, axis=1)
df_test['nquerys_in_title'] = df_test[['product_title', 'search_term']].apply(query_counter, axis=1)

# feature #11: count how many times query appeared in description
df_train['nquerys_in_description'] = df_train[['product_description', 'search_term']].apply(query_counter, axis=1)
df_test['nquerys_in_description'] = df_test[['product_description', 'search_term']].apply(query_counter, axis=1)

# feature #12: nwords_in_title/nwords_title
df_train['ratio_nwords_in_title'] = df_train['nwords_in_title'] / df_train['nwords_title']
df_test['ratio_nwords_in_title'] = df_test['nwords_in_title'] / df_test['nwords_title']

# feature #13: nwords_in_description/nwords_description
df_train['ratio_nwords_in_description'] = df_train['nwords_in_description'] / df_train['nwords_description']
df_test['ratio_nwords_in_description'] = df_test['nwords_in_description'] / df_test['nwords_description']

# feature #14: nquerys_in_title/nwords_title
df_train['ratio_nquerys_in_title'] = df_train['nquerys_in_title'] / df_train['nwords_title']
df_test['ratio_nquerys_in_title'] = df_test['nquerys_in_title'] / df_test['nwords_title']

# feature #15: nquerys_in_description/nwords_description
df_train['ratio_nquerys_in_description'] = df_train['nquerys_in_description'] / df_train['nwords_description']
df_test['ratio_nquerys_in_description'] = df_test['nquerys_in_description'] / df_test['nwords_description']

# feature #16: how many word in brand hit query words
df_train['nwords_in_brand'] = df_train[['brand', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_brand'] = df_test[['brand', 'search_term']].apply(words_counter, axis=1)

# feature #17: nwords_in_brand/nwords_search
df_train['ratio_nwords_in_brand'] = df_train['nwords_in_brand'] / df_train['nwords_search']
df_test['ratio_nwords_in_brand'] = df_test['nwords_in_brand'] / df_test['nwords_search']

# feature #18 how many seach word in brand name
df_train['nwords_in_query_hit_brand'] = df_train[['brand', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_brand'] = df_test[['brand', 'search_term']].apply(words_counter_in_query, axis=1)

# feature #19: how many word in material hit query words
df_train['nwords_in_material'] = df_train[['material', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_material'] = df_test[['material', 'search_term']].apply(words_counter, axis=1)

# feature #20: nwords_in_material/nwords_search
df_train['ratio_nwords_in_material'] = df_train['nwords_in_material'] / df_train['nwords_search']
df_test['ratio_nwords_in_material'] = df_test['nwords_in_material'] / df_test['nwords_search']

# feature #21 how many seach word in material
df_train['nwords_in_query_hit_material'] = df_train[['material', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_material'] = df_test[['material', 'search_term']].apply(words_counter_in_query, axis=1)

# feature #22: how many word in color hit query words
df_train['nwords_in_color'] = df_train[['color', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_color'] = df_test[['color', 'search_term']].apply(words_counter, axis=1)

# feature #23: nwords_in_color/nwords_search
df_train['ratio_nwords_in_color'] = df_train['nwords_in_color'] / df_train['nwords_search']
df_test['ratio_nwords_in_color'] = df_test['nwords_in_color'] / df_test['nwords_search']

# feature #24 how many seach word in color
df_train['nwords_in_query_hit_color'] = df_train[['color', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_color'] = df_test[['color', 'search_term']].apply(words_counter_in_query, axis=1)

# feature #25: how many word in bullet hit query words
df_train['nwords_in_bullet'] = df_train[['bullet', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_bullet'] = df_test[['bullet', 'search_term']].apply(words_counter, axis=1)

# feature #26: nwords_in_bullet/nwords_search
df_train['ratio_nwords_in_bullet'] = df_train['nwords_in_bullet'] / df_train['nwords_search']
df_test['ratio_nwords_in_bullet'] = df_test['nwords_in_bullet'] / df_test['nwords_search']

# feature #27 how many seach word in bullet
df_train['nwords_in_query_hit_bullet'] = df_train[['bullet', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_bullet'] = df_test[['bullet', 'search_term']].apply(words_counter_in_query, axis=1)

# feature #28: how many word in name hit query words
df_train['nwords_in_name'] = df_train[['name', 'search_term']].apply(words_counter, axis=1)
df_test['nwords_in_name'] = df_test[['name', 'search_term']].apply(words_counter, axis=1)

# feature #29: nwords_in_name/nwords_search
df_train['ratio_nwords_in_name'] = df_train['nwords_in_name'] / df_train['nwords_search']
df_test['ratio_nwords_in_name'] = df_test['nwords_in_name'] / df_test['nwords_search']

# feature #30 how many seach word in name
df_train['nwords_in_query_hit_name'] = df_train[['name', 'search_term']].apply(words_counter_in_query, axis=1)
df_test['nwords_in_query_hit_name'] = df_test[['name', 'search_term']].apply(words_counter_in_query, axis=1)

print 'done, {}'.format(datetime.now() - start)

columns_drop = [u'id', u'product_uid', u'product_title', u'search_term', u'product_description', u'brand', u'material', u'color', u'bullet', u'name', u'tfidf_title', u'tfidf_description']

df_train.drop(labels=columns_drop, axis=1, inplace=True)
df_test.drop(labels=columns_drop, axis=1, inplace=True)

with open(PROJECT_PATH + 'pickles/df_features.pkl', 'w') as f:
    pickle.dump((df_train, df_test), f)