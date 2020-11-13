import torch
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
import swifter
from scipy.spatial import distance


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)

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

def get_distance(text1:str, text2:str) -> float:
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)['last_hidden_state'].detach().numpy()

    ret = 0
    step_ct = 0

    for t1, t2 in zip(outputs[0], outputs[1]):
        ret += distance.euclidean(t1, t2)
        step_ct += 1

    return ret/step_ct

def get_distance_wrapper(terms:pd.Series, titles:pd.Series) -> list:
    ret = []
    assert len(terms) == len(titles)
    n:int = len(terms)
    t:int = 0
    print("Start Calculating Euclidean Distances: ")
    print("Progress: {}/{}".format(t,n), end='\r', flush=True)
    for t1, t2 in zip(terms.tolist(), titles.tolist()):
        ret.append(get_distance(t1, t2))
        t += 1
        print("Progress: {}/{}".format(t,n), end='\r', flush=True)
    
    return ret



def main():
    train_raw, test_raw = _preprocessing()

    # train_raw, test_raw = train_raw[:100], test_raw[:100]

    term_title_train = get_distance_wrapper(train_raw['search_term'], train_raw['product_title'])
    term_title_test = get_distance_wrapper(test_raw['search_term'], test_raw['product_title'])

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
