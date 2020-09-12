# Python 3.7
# This file shall not be imported to another Python Interpreter due to multiprocessing package.

import numpy as np
import pandas as pd
import itertools
import multiprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import pickle


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
train_val_X = train_val_X[idx]
train_val_Y = train_val_Y[idx]

# Data Recording
result_log = {'RegularTree': None,
           'RandomForest': None,
           'XGBoost': None}
best_para = {'RegularTree': None,
             'RandomForest': None,
             'XGBoost': None}





############## Regular Decision Tree #####################
print("***********Training Regular Decision Tree****************")
print("\n")
print("*********************Job 2a******************************")


###### Job 2a: plot figure on training error and validation error wrt criterion #####
# Train-Val split
ratio = 0.8
split_index = int(ratio * len(train_val_Y))

train_X = train_val_X[:split_index]
val_X = train_val_X[split_index:]

train_Y = train_val_Y[:split_index]
val_Y = train_val_Y[split_index:]

print("------------------------------------------------------")
print("{}% of data are used for training.".format(int(ratio * 100)))
print("Number of training samples: {}".format(len(train_Y)))
print("Number of validation samples: {}".format(len(val_Y)))
print("Number of test samples: {}".format(len(test_raw)))
print("------------------------------------------------------")
print("------------------------------------------------------")
print("Shape of TfidfVectorizer Matrix: {}".format(train_X.shape))
print("------------------------------------------------------")

# Tunable parameters
# depths = [10, 25, 50, 75, 100, 125, 150]
# ccp_alpha = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
criteria = ['gini', 'entropy']

print("*******Training trees for Job 2a")
train_errs = []
val_errs = []
for c in criteria:
    print("    Current criterion: ", c)
    dtc = tree.DecisionTreeClassifier(max_depth=150, criterion=c)
    dtc.fit(train_X, train_Y)
    train_errs.append(1 - dtc.score(train_X, train_Y))
    val_errs.append(1 - dtc.score(val_X, val_Y))

print("Saving Plot....")
plt.plot(criteria, train_errs, marker='.', label="Training Error")
plt.plot(criteria, val_errs, marker='.', label="Validation Error")
plt.xlabel('Criteria')
plt.ylabel('Error')
plt.title('Job 2a: Regular tree error VS criterion')
plt.legend()
plt.savefig('./submissions/2a.png')

# Free memory
del train_errs
del val_errs
del dtc
del train_X
del val_X
del train_Y
del val_Y



########## Job 2b: Cross validate regular tree #################
print("\n")
print("*********************Job 2b******************************")
criteria = ['gini', 'entropy']
min_samples_leaves = [1, 2, 4, 8, 16, 32]
max_features = [600000, 400000, 200000, 100000, 50000, 25000, 10000, 5000, 2500, 1000, 500, 100]


def trainRegularTreeCrossValidation(crit, leaf, feature):
    train_err_list = []
    val_err_list = []
    _kf = KFold(n_splits=5)
    for _train_index, _val_index in _kf.split(train_val_X):
        _X = train_val_X[_train_index, :]
        _X_val = train_val_X[_val_index, :]
        _Y = train_val_Y[_train_index]
        _Y_val = train_val_Y[_val_index]

        _dtc = tree.DecisionTreeClassifier(criterion=crit, min_samples_leaf=leaf,
                                          max_features=feature)
        _dtc.fit(_X, _Y)
        train_err_list.append(1 - _dtc.score(_X, _Y))
        val_err_list.append(1 - _dtc.score(_X_val, _Y_val))

    _avg_train_err = sum(train_err_list) / len(train_err_list)
    _avg_val_err = sum(val_err_list) / len(val_err_list)

    return crit, leaf, feature, _avg_train_err, _avg_val_err


with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    args = list(itertools.product(criteria, min_samples_leaves, max_features))

    results = p.starmap(trainRegularTreeCrossValidation, args)
    results_df = pd.DataFrame(results,
                              columns=('crit', 'leaf', 'feature', 'avg_train_err', 'avg_val_err'))
    result_log['RegularTree'] = results_df

    # Find best combination
    best_row_id = results_df['avg_val_err'].argmin()
    best_row = results_df.iloc[best_row_id]

    best_crit, best_leaf, best_feature = best_row['crit'], best_row['leaf'], best_row['feature']
    best_para['RegularTree'] = {'criterion': best_crit,
                                'min_samples_leaf': str(best_leaf),
                                'max_features': str(best_feature)}


# Plot Figures for submission
for para in ['crit', 'leaf', 'feature']:
    _df = results_df[[para, 'avg_train_err', 'avg_val_err']].groupby(para).mean()
    plt.clf()
    plt.plot(_df['avg_train_err'].index, _df['avg_train_err'], marker='.', label='Average Training Error')
    plt.plot(_df['avg_val_err'].index, _df['avg_val_err'], marker='.', label='Average Validation Error')
    plt.legend()
    if para == 'crit':
        plt.xlabel('criterion')
        plt.title('Decision tree error VS criterion')
    if para == 'leaf':
        plt.xlabel('min_samples_leaf')
        plt.title('Decision tree error VS min_samples_leaf')
    if para == 'feature':
        plt.xlabel('max_features')
        plt.title('Decision tree error VS max_features')
    plt.ylabel('Average Error')

    plt.savefig('./submissions/2b_{}.png'.format(para))





########################### Random Forest ###########################
print("**********************Start training random forest model********************")

# Use best_crit, best_leaf, best_feature
n_esti = [25, 50, 75, 100, 125, 150, 200, 300]


def trainRandomForestCrossValidation(n):
    train_acc_list = []
    val_acc_list = []
    _kf = KFold(n_splits=5)
    for _train_index, _val_index in _kf.split(train_val_X):
        _X = train_val_X[_train_index, :]
        _X_val = train_val_X[_val_index, :]
        _Y = train_val_Y[_train_index]
        _Y_val = train_val_Y[_val_index]

        _rndf = RandomForestClassifier(n_estimators=n,
                                       criterion=best_crit,
                                       min_samples_leaf=best_leaf,
                                       max_features=best_feature)
        _rndf.fit(_X, _Y)
        train_acc_list.append(_rndf.score(_X, _Y))
        val_acc_list.append(_rndf.score(_X_val, _Y_val))

    _avg_train_acc = sum(train_acc_list) / len(train_acc_list)
    _avg_val_acc = sum(val_acc_list) / len(val_acc_list)

    _std_train_acc = np.std(train_acc_list)
    _std_val_acc = np.std(val_acc_list)

    return n, _avg_train_acc, _avg_val_acc, _std_train_acc, _std_val_acc


with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    results = p.map(trainRandomForestCrossValidation, n_esti)
    results_df = pd.DataFrame(results,
                              columns=('n', 'avg_train_acc', 'avg_val_acc',
                                       'std_train_acc', 'std_val_acc'))
    result_log['RandomForest'] = results_df

    # Find best combination
    best_row_id = results_df['avg_val_acc'].argmax()
    best_row = results_df.iloc[best_row_id]

    best_n = best_row['n']
    best_para['RandomForest'] = {'Best_n_estimator': str(best_n)}



############ Plot figure ############
print('Plot Figure....')
plt.clf()
plt.plot(n_esti, results_df['avg_train_acc'], marker='.', label="Average Training Accuracy")
plt.plot(n_esti, results_df['avg_val_acc'], marker='.', label="Average Validation Accuracy")
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('Average Accuracy')
plt.title('Random forest accuracy VS Number of trees')
plt.savefig('./submissions/3d.png')

#STD
plt.clf()
plt.plot(n_esti, results_df['std_train_acc'], marker='.', label="Training Accuracy Standard Deviation")
plt.plot(n_esti, results_df['std_val_acc'], marker='.', label="Validation Accuracy Standard Deviation")
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy Standard Deviation')
plt.title('Random forest Accuracy Standard Deviation VS Number of trees')
plt.savefig('./submissions/3d_std.png')






################ XGBoost ###############
print('***************** Train XGBoost Model *******************')


lrs = [0.001, 0.01, 0.1, 0.15, 0.2, 0.3]
train_acc_all = []
val_acc_all = []

train_acc_std = []
val_acc_std = []

for lr in lrs:
    train_acc_list = []
    val_acc_list = []
    kf = KFold(n_splits=5)
    for train_index, val_index in kf.split(train_val_X):
        X = train_val_X[train_index, :]
        X_val = train_val_X[val_index, :]
        Y = train_val_Y[train_index]
        Y_val = train_val_Y[val_index]

        clf = xgb.XGBClassifier(n_estimators=int(best_n), learning_rate=lr, nthread=multiprocessing.cpu_count())

        clf.fit(X, Y)
        train_acc_list.append(clf.score(X, Y))
        val_acc_list.append(clf.score(X_val, Y_val))

    train_acc_all.append(sum(train_acc_list) / len(train_acc_list))
    val_acc_all.append(sum(val_acc_list) / len(val_acc_list))
    train_acc_std.append(np.std(train_acc_list))
    val_acc_std.append(np.std(val_acc_list))

results_df = pd.DataFrame(zip(lrs, train_acc_all, val_acc_all, train_acc_std, val_acc_std),
                          columns=('lr', 'avg_train_acc', 'avg_val_acc',
                                   'train_acc_std', 'val_acc_std'))
result_log['XGBoost'] = results_df

# Find best combination
best_row_id = results_df['avg_val_acc'].argmax()
best_row = results_df.iloc[best_row_id]

best_lr = best_row['lr']
best_para['XGBoost'] = {'Best_lr': str(best_lr)}


############ Plot figure ############
print('Plot Figure....')
plt.clf()
plt.plot(lrs, results_df['avg_train_acc'], marker='.', label="Training Accuracy Standard Deviation")
plt.plot(lrs, results_df['avg_val_acc'], marker='.', label="Validation Accuracy Standard Deviation")
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy Standard Deviation')
plt.title('XGBoost Accuracy Standard Deviation VS Learning Rate')
plt.savefig('./submissions/4d_std.png')

#STD
plt.clf()
plt.plot(lrs, results_df['std_train_acc'], marker='.', label="Average Training Accuracy")
plt.plot(lrs, results_df['std_val_acc'], marker='.', label="Average Validation Accuracy")
plt.legend()
plt.xlabel('Learning Rate')
plt.ylabel('Average Accuracy')
plt.title('XGBoost Accuracy VS Learning Rate')
plt.savefig('./submissions/4d.png')


# Save Results and best parameters
with open('./submissions/best_para.json', 'w') as f:
    json.dump(best_para, f)

with open('./submissions/results.pkl', 'wb') as f:
    pickle.dump(result_log, f)
