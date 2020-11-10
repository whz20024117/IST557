import pandas as pd
import swifter
from nltk.stem import SnowballStemmer
from nltk import RegexpTokenizer


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


# Save
train_all.to_csv('./train_all.csv', index=False)
test_all.to_csv('./test_all.csv', index=False)
