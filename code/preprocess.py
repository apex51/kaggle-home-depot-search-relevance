'''
the target columns now are:

x: ['search_term', 'product_title', 'product_description', 'brand']
y: ['relevance']

'''

import numpy as np
import pandas as pd
import pickle
import re
from datetime import datetime

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_PATH = '/Users/jianghao/Projects/mine/home_depot/'

df_train = pd.read_csv(PROJECT_PATH + 'data/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv(PROJECT_PATH + 'data/test.csv', encoding='ISO-8859-1')
df_description = pd.read_csv(PROJECT_PATH + 'data/product_descriptions.csv')
df_attribute = pd.read_csv(PROJECT_PATH + 'data/attributes.csv', encoding='ISO-8859-1').dropna()
# brand feature
df_brand = df_attribute[df_attribute['name'] == 'MFG Brand Name'][['product_uid', 'value']].rename(columns={'value':'brand'})
# material feature
df_material = df_attribute[df_attribute['name'].apply(lambda x: 'material' in x.lower())].groupby('product_uid')
df_material = df_material['value'].apply(func=lambda x: ' '.join(x)).reset_index().rename(columns={'value':'material'})
# color feature
df_color = df_attribute[df_attribute['name'].apply(lambda x: 'color' in x.lower())].groupby('product_uid')
df_color = df_color['value'].apply(func=lambda x: ' '.join(x)).reset_index().rename(columns={'value':'color'})
# bulletX feature
df_bullet = df_attribute[df_attribute['name'].apply(lambda x: 'bullet' in x.lower())].groupby('product_uid')
df_bullet = df_bullet['value'].apply(func=lambda x: ' '.join(x)).reset_index().rename(columns={'value':'bullet'})
# feat_names feature
df_names = df_attribute[['product_uid', 'name']].groupby('product_uid')
df_names = df_names['name'].apply(func=lambda x: ' '.join(x)).reset_index()

# merge feats
# description
df_train = pd.merge(df_train, df_description, how='left', on='product_uid')
df_test = pd.merge(df_test, df_description, how='left', on='product_uid')
# brand
df_train = pd.merge(df_train, df_brand, how='left', on='product_uid')
df_test = pd.merge(df_test, df_brand, how='left', on='product_uid')
# material
df_train = pd.merge(df_train, df_material, how='left', on='product_uid')
df_test = pd.merge(df_test, df_material, how='left', on='product_uid')
# color
df_train = pd.merge(df_train, df_color, how='left', on='product_uid')
df_test = pd.merge(df_test, df_color, how='left', on='product_uid')
# bulletX
df_train = pd.merge(df_train, df_bullet, how='left', on='product_uid')
df_test = pd.merge(df_test, df_bullet, how='left', on='product_uid')
# feat_names
df_train = pd.merge(df_train, df_names, how='left', on='product_uid')
df_test = pd.merge(df_test, df_names, how='left', on='product_uid')


# perform spell check
import google_spell_checker

n_changed = [0]

def spell_checker(search_terms, count_change):
    keys = google_spell_checker.spell_check_dict.keys()
    if search_terms in keys:
        count_change[0] += 1
        return google_spell_checker.spell_check_dict[search_terms]
    else:
        return search_terms

print '='*20
print 'checking spelling for queries...'
start = datetime.now()
df_train['search_term'] = df_train['search_term'].apply(lambda x: spell_checker(x, n_changed)) # 10076 terms changed in train
df_test['search_term'] = df_test['search_term'].apply(lambda x: spell_checker(x, n_changed)) # 17736 terms changed in test
print 'done, {}'.format(datetime.now() - start)


# perform stemming
stemmer = SnowballStemmer('english')
vectorizer = CountVectorizer()
analyzer = vectorizer.build_analyzer()
# convenience func for stemming string
# preprocessing text
def str_stemmer(s):
    if isinstance(s, str) or isinstance(s, unicode):
        s = re.sub(r"([a-z])(\.?)([A-Z])([a-z])", r"\1\2 \3\4", s) # sepreate 'above.California' like words 
        s = s.lower()
        # s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+\.?[0-9]*)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")
        s = s.replace("  "," ")
        s = (' ').join([stemmer.stem(word) for word in re.split(r'[.,!?;:"\s\(\)-]', s)])
        return s.lower()
    else:
        return ''

print '='*20
print 'stemming...'
start = datetime.now()
df_train['search_term'] = df_train['search_term'].apply(str_stemmer)
df_train['product_title'] = df_train['product_title'].apply(str_stemmer)
df_train['product_description'] = df_train['product_description'].apply(str_stemmer)
df_test['search_term'] = df_test['search_term'].apply(str_stemmer)
df_test['product_title'] = df_test['product_title'].apply(str_stemmer)
df_test['product_description'] = df_test['product_description'].apply(str_stemmer)

# process 'brand' column
df_train['brand'] = df_train['brand'].apply(str_stemmer)
df_test['brand'] = df_test['brand'].apply(str_stemmer)

# process 'material' column
df_train['material'] = df_train['material'].apply(str_stemmer)
df_test['material'] = df_test['material'].apply(str_stemmer)

# process 'color' column
df_train['color'] = df_train['color'].apply(str_stemmer)
df_test['color'] = df_test['color'].apply(str_stemmer)

# process 'bullet' column
df_train['bullet'] = df_train['bullet'].apply(str_stemmer)
df_test['bullet'] = df_test['bullet'].apply(str_stemmer)

# process 'name' column
df_train['name'] = df_train['name'].apply(str_stemmer)
df_test['name'] = df_test['name'].apply(str_stemmer)

print 'done, {}'.format(datetime.now() - start)


with open(PROJECT_PATH + 'pickles/df_preprocessed.pkl', 'wb') as f:
    pickle.dump((df_train, df_test), f)

######################################
# generate stratified 5-fold 
######################################

from sklearn.cross_validation import StratifiedKFold

y_train = df_train['relevance'].copy()

# transform minor values
y_train[y_train == 2.75] = 2.67
y_train[y_train == 2.50] = 2.33
y_train[y_train == 2.25] = 2.33
y_train[y_train == 1.75] = 1.67
y_train[y_train == 1.50] = 1.33
y_train[y_train == 1.25] = 1.33

# generate stratified 5-fold
sk5f = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=22)

with open(PROJECT_PATH + 'pickles/df_preprocessed_sk5f.pkl', 'wb') as f:
    pickle.dump(sk5f, f)

























