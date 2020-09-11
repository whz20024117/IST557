import numpy as np
import pandas as pd
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
import pickle


with open('./submissions/best_para.json') as f:
    best_para = json.load(f)

with open('./submissions/results.pkl', 'rb') as f:
    result_log = pickle.load(f)

print("Print best result each model can achieve: ")
best_err_reg = result_log['RegularTree']['avg_val_err'].min()
print("    Error rate using RegularTree: {:.4f}".format(best_err_reg))
best_err_rndf = result_log['RandomForest']['avg_val_err'].min()
print("    Error rate using RandomForest: {:.4f}".format(best_err_rndf))
best_err_xgb = result_log['XGBoost']['avg_val_err'].min()
print("    Error rate using XGBoost: {:.4f}".format(best_err_xgb))

# Using XGBoost
import xgboost as xgb
para_xgb = {'lr': best_para['XGBoost']['Best_lr'],
            'n': int(float(best_para['RandomForest']['Best_n_estimator']))}

clf = xgb.XGBClassifier(n_estimators=para_xgb['n'],
                        learning_rate=para_xgb['lr'],
                        nthread=multiprocessing.cpu_count())

# Load Data
train_val_df = pd.read_csv('./news-train.csv')
train_val_raw = train_val_df['Text'].tolist()
train_val_label = train_val_df['Category'].tolist()
test_df = pd.read_csv('./news-test.csv')
test_raw = test_df['Text'].tolist()

le = LabelEncoder()
train_val_Y = le.fit_transform(train_val_label)

# Tfdif vectorizer (with uni, bi, and tri gram)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
train_val_X = vectorizer.fit_transform(train_val_raw)
test_X = vectorizer.transform(test_raw)

# Shuffle
np.random.seed(0)
idx = np.random.permutation(train_val_X.shape[0])
X = train_val_X[idx]
Y = train_val_Y[idx]


# Train
clf.fit(X, Y)
# Test
print("Error rate on training set: ", 1 - clf.score(X, Y))
test_label = clf.predict(test_X)

label_csv = pd.DataFrame(zip(test_df['ArticleId'].tolist(), le.inverse_transform(test_label)))
label_csv.to_csv('./submissions/labels.csv', header=None, index=None)
