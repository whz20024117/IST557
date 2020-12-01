import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import swifter
from nltk.stem import SnowballStemmer
from nltk import RegexpTokenizer
import json
import torch
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from scipy.spatial import distance


with open('spellCheck.json', 'r') as f:
    spell_check_dict = json.load(f)


# import tensorflow as tf

# # Set CPU as available physical device
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')



def fixSpelling(term):
    try:
        return spell_check_dict[term]
    except KeyError:
        return term

def _preprocessing():

    ############### Start with data processing ####################
    print("Start Loading and Processing Raw Data....")
    attributes = pd.read_csv('data/attributes.csv.zip')
    # JESUS who knows fking Kaggle data has Null... LOL
    attributes = attributes.dropna()
    attributes = attributes.astype({'product_uid': int, 'name':str, 'value':str})
    descriptions = pd.read_csv('data/product_descriptions.csv.zip', dtype={'product_uid': int, 'product_description':str})


    traintest_dtype_dict = {'product_uid': int, 'product_title': str, 'search_term': str, 'relevance': float}
    # Literally... this stupid af West European encoding, Kaggle needs to work harder on their data
    train = pd.read_csv('data/train.csv.zip', encoding='latin_1', dtype=traintest_dtype_dict)
    test = pd.read_csv('data/test.csv.zip', encoding='latin_1', dtype=traintest_dtype_dict)

    # Make one attribute per product
    attributes['attr'] = attributes['name'] + ' ' + attributes['value']
    attributes = attributes[['product_uid', 'attr']].groupby(['product_uid'])['attr'].apply(' '.join).reset_index()

    # Merge all the data
    train_all = pd.merge(train, descriptions, how='left', on='product_uid')
    train_all = pd.merge(train_all, attributes, how='left', on='product_uid').fillna('')

    test_all = pd.merge(test, descriptions, how='left', on='product_uid')
    test_all = pd.merge(test_all, attributes, how='left', on='product_uid').fillna('')

    ####################### Spell Check #########################
    train_all['search_term'] = train_all['search_term'].swifter.apply(fixSpelling)

    return train_all, test_all



def stem(train_all, test_all):
    ################### Feature Making ###################
    print("Stem strings...")
    stemmer = SnowballStemmer('english')
    puncRemove = RegexpTokenizer(r"\w+")


    def textProcess(text):
        tokens = puncRemove.tokenize(text)
        tokens = [stemmer.stem(w) for w in tokens]
        return ' '.join(tokens)


    for str_col in ['product_title', 'search_term', 'product_description', 'attr']:
        train_all[str_col] = train_all[str_col].swifter.apply(textProcess)
        test_all[str_col] = test_all[str_col].swifter.apply(textProcess)


    train_all.to_csv('./train_all_ckpt3.csv', index=False)
    test_all.to_csv('./test_all_ckpt3.csv', index=False)



def bertSimilarity(train_raw, test_raw):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    pool = model.start_multi_process_pool()

    train_term_emb = model.encode_multi_process(train_raw['search_term'].tolist(), pool)
    train_title_emb = model.encode_multi_process(train_raw['product_title'].tolist(), pool)
    train_description_emb = model.encode_multi_process(train_raw['product_description'].tolist(), pool)

    test_term_emb = model.encode_multi_process(test_raw['search_term'].tolist(), pool)
    test_title_emb = model.encode_multi_process(test_raw['product_title'].tolist(), pool)
    test_description_emb = model.encode_multi_process(test_raw['product_description'].tolist(), pool)

    distance_names = ["euc", "man", "bray", 
                        "can", "chebyshev", "corr", "min"]
    distance_funcs = [distance.euclidean, distance.cityblock, distance.braycurtis, 
                        distance.canberra, distance.chebyshev, distance.correlation, distance.minkowski]
    
    # New dataframe
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df['id'] = train_raw['id']
    test_df['id'] = test_raw['id']
    
    for dname, dfunc in zip(distance_names, distance_funcs):
        term_title_train = [dfunc(t1, t2) for t1, t2 in zip(train_term_emb, train_title_emb)]
        term_title_test = [dfunc(t1, t2) for t1, t2 in zip(test_term_emb, test_title_emb)]

        train_df['title_bert_' + dname] = term_title_train
        test_df['title_bert_' + dname] = term_title_test

        term_description_train = [dfunc(t1, t2) for t1, t2 in zip(train_term_emb, train_description_emb)]
        term_description_test = [dfunc(t1, t2) for t1, t2 in zip(test_term_emb, test_description_emb)]

        train_df['description_bert_' + dname] = term_description_train
        test_df['description_bert_' + dname] = term_description_test


    train_df.to_csv('./train_dist_bert.csv', index=False)
    test_df.to_csv('./test_dist_bert.csv', index=False)


def tfSimilarity(train_raw, test_raw):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    train_term_emb = embed(train_raw['search_term'].tolist())
    train_title_emb = embed(train_raw['product_title'].tolist())
    train_description_emb = embed(train_raw['product_description'].tolist())

    test_term_emb = embed(test_raw['search_term'].tolist())
    test_title_emb = embed(test_raw['product_title'].tolist())
    test_description_emb = embed(test_raw['product_description'].tolist())

    distance_names = ["euc", "man", "bray", 
                        "can", "chebyshev", "corr", "min"]
    distance_funcs = [distance.euclidean, distance.cityblock, distance.braycurtis, 
                        distance.canberra, distance.chebyshev, distance.correlation, distance.minkowski]
    
    # New dataframe
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df['id'] = train_raw['id']
    test_df['id'] = test_raw['id']
    
    for dname, dfunc in zip(distance_names, distance_funcs):
        term_title_train = [dfunc(t1, t2) for t1, t2 in zip(train_term_emb, train_title_emb)]
        term_title_test = [dfunc(t1, t2) for t1, t2 in zip(test_term_emb, test_title_emb)]

        train_df['title_tf_' + dname] = term_title_train
        test_df['title_tf_' + dname] = term_title_test

        term_description_train = [dfunc(t1, t2) for t1, t2 in zip(train_term_emb, train_description_emb)]
        term_description_test = [dfunc(t1, t2) for t1, t2 in zip(test_term_emb, test_description_emb)]

        train_df['description_tf_' + dname] = term_description_train
        test_df['description_tf_' + dname] = term_description_test


    train_df.to_csv('./train_dist_tf.csv', index=False)
    test_df.to_csv('./test_dist_tf.csv', index=False)



if __name__ == "__main__":
    train_all, test_all = _preprocessing()

    # Similarity use universal sentence encoder
    tfSimilarity(train_all, test_all)
    
    # Similarity before stem.. since it needs raw text...
    bertSimilarity(train_all, test_all)
    
    # All task require raw text should be done before calling stem
    stem(train_all, test_all)
    