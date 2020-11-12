import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from scipy.spatial import distance


def _preprocessing():
    # Code from preprocessing.py

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

    return train_all, test_all



def main():
    train_raw, test_raw = _preprocessing()

    # train_raw, test_raw = train_raw[:100], test_raw[:100]

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    pool = model.start_multi_process_pool()

    train_term_emb = model.encode_multi_process(train_raw['search_term'].tolist(), pool)
    train_title_emb = model.encode_multi_process(train_raw['product_title'].tolist(), pool)

    test_term_emb = model.encode_multi_process(test_raw['search_term'].tolist(), pool)
    test_title_emb = model.encode_multi_process(test_raw['product_title'].tolist(), pool)

    term_title_train = [distance.euclidean(t1, t2) for t1, t2 in zip(train_term_emb, train_title_emb)]
    term_title_test = [distance.euclidean(t1, t2) for t1, t2 in zip(test_term_emb, test_title_emb)]

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df['id'] = train_raw['id']
    test_df['id'] = test_raw['id']
    train_df['title_dist'] = term_title_train
    test_df['title_dist'] = term_title_test

    train_df.to_csv('./train_dist.csv', index=False)
    test_df.to_csv('./test_dist.csv', index=False)


if __name__ == '__main__':
    main()
