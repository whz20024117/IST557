import numpy as np
import pandas as pd
import itertools
import multiprocessing
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import graphviz


train_val_df = pd.read_csv('./news-train.csv')
train_val_raw = train_val_df['Text'].tolist()
train_val_label = train_val_df['Category'].tolist()
test_df = pd.read_csv('./news-test.csv')
test_raw = test_df['Text'].tolist()

le = LabelEncoder()
train_val_Y = le.fit_transform(train_val_label)

# Count vectorizer (with uni, bi, and tri gram)
ct_vectorizer = CountVectorizer(ngram_range=(1, 3))
ct_train_val_X = ct_vectorizer.fit_transform(train_val_raw)
ct_test_X = ct_vectorizer.transform(test_raw)

# Tfdif vectorizer (with uni, bi, and tri gram)
tf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tf_train_val_X = tf_vectorizer.fit_transform(train_val_raw)
tf_test_X = tf_vectorizer.transform(train_val_raw)

# Shuffle
np.random.seed(0)
idx = np.random.permutation(tf_train_val_X.shape[0])
ct_train_val_X = ct_train_val_X[idx]
tf_train_val_X = tf_train_val_X[idx]
train_val_Y = train_val_Y[idx]