import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

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
 filtered = [w for w in tokens if not w in stopwords.words('english')]
 # stemming process
 stemmed = []
 for item in filtered:
   stemmed.append(stemmer.stem(item))
 # final unigrams
 return stemmed
