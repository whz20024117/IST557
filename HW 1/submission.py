import string
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import json
import operator


#################### Sample Unigram Code ######################

# stemming tool from nltk
stemmer = PorterStemmer()
# a mapping dictionary that help remove punctuations
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def get_tokens(text):
    # turn document into lowercase
    lowers = text.lower()
    # remove punctuations
    no_punctuation = lowers.translate(remove_punctuation_map)
    # tokenize document
    tokens = nltk.word_tokenize(no_punctuation)
    # remove stop words
    filtered = [w for w in tokens if w not in stopwords.words('english')]
    # stemming process
    stemmed = []
    for item in filtered:
        stemmed.append(stemmer.stem(item))
    # final unigrams
    return stemmed

##############################################################################


with open('./dictionary.txt', 'r') as f:
    word_dict = [line.rstrip() for line in f]

news = pd.read_csv('./news-train.csv')
unigram = [get_tokens(s) for s in news['Text'].tolist()]
unigram_dict = [[w for w in doc if w in word_dict] for doc in unigram]

# Calculate TFIDF
mat = np.zeros(shape=(len(unigram_dict), len(word_dict)))
assert mat.shape == (1490, 1000)
m = [sum([1 if w in doc else 0 for doc in unigram_dict]) for w in word_dict]
for i in range(mat.shape[0]):
    freq = [unigram_dict[i].count(w) for w in word_dict]
    for j in range(mat.shape[1]):
        tf = unigram_dict[i].count(word_dict[j]) / max(freq)
        idf = np.log(len(unigram_dict)/(m[j]))
        mat[i][j] = tf * idf

# Save TFIDF matrix
np.savetxt('matrix.txt', mat, fmt='%.6f', delimiter=',')

# Top 3 most freq words in each category
cat = news['Category'].tolist()
word_freq = {}
for c in set(cat):
    unigram_dict_cat = [doc for doc, k in zip(unigram_dict, cat) if k == c]
    freq = [sum([doc.count(w) for doc in unigram_dict_cat]) for w in word_dict]
    freq_mapping = {key: value for key, value in zip(word_dict, freq)}
    top3 = dict(sorted(freq_mapping.items(), key=operator.itemgetter(1), reverse=True)[:3])
    word_freq[c] = top3


with open('frequency.json', 'w') as f:
    json.dump(word_freq, f)

# Top 3 highest tfidf words in each category
word_tfidf = {}
for c in set(cat):
    tfidf_cat = np.array([tfidf for tfidf, k in zip(mat.tolist(), cat) if k == c])
    mean_tfidf_cat = tfidf_cat.mean(axis=0)
    tfidf_mapping = {key: value for key, value in zip(word_dict, mean_tfidf_cat)}
    top3 = dict(sorted(tfidf_mapping.items(), key=operator.itemgetter(1), reverse=True)[:3])
    word_tfidf[c] = top3

with open('scores.json', 'w') as f:
    json.dump(word_tfidf, f)
