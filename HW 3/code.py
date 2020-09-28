import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, plot_roc_curve
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt



# Helper functions
def _load_shuffle_data(path, seed=0):
    df = pd.read_csv(path)
    X = df.drop('Class', axis=1).to_numpy(dtype=np.float32)
    y = df['Class'].to_numpy(dtype=np.float32)

    # Shuffle Data
    np.random.seed(seed)
    idx = np.random.permutation(range(len(X)))

    return X[idx], y[idx]


def _auc_training_process(clf_class, X, y, **kwargs):
    """
    :param clf_class: sklearn-style classifier class
    :param X: training data
    :param y: label
    :param kwargs: arguments pass to classifier class __init__
    :return: list: rain_auc_scores, list: auc_scores
    """

    # 5 fold cv
    kf = KFold(n_splits=5)

    auc_scores = []

    for i_train, i_test in kf.split(X):
        train_X, train_y = X[i_train], y[i_train]
        test_X, test_y = X[i_test], y[i_test]

        clf = clf_class(**kwargs)
        clf.fit(train_X, train_y)

        test_auc = roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1])
        auc_scores.append(test_auc)

    return auc_scores


def _svm_training_process(X, y, C):
    """
    :param clf_class: sklearn-style classifier class
    :param X: training data
    :param y: label
    :param kwargs: arguments pass to classifier class __init__
    :return: list: rain_auc_scores, list: auc_scores
    """

    # 5 fold cv
    kf = KFold(n_splits=5)

    auc_scores = []

    for i_train, i_test in kf.split(X):
        train_X, train_y = X[i_train], y[i_train]
        test_X, test_y = X[i_test], y[i_test]

        clf = make_pipeline(StandardScaler(), SVC(C=C, probability=True))
        clf.fit(train_X, train_y)

        test_auc = roc_auc_score(test_y, clf.predict_proba(test_X)[:, 1])
        auc_scores.append(test_auc)

    return auc_scores


def _print_results(auc_scores):
    mean = np.mean(auc_scores)
    std = np.std(auc_scores)

    print("        Average test AUC score: {:.4f}".format(mean))
    print("        Standard Deviation of test AUC: {:.4f}".format(std))


def _save_results(path, auc_scores, p_name=None, p=None):
    mean = np.mean(auc_scores, axis=-1)
    std = np.std(auc_scores, axis=-1)

    if p_name:
        assert len(p) == len(auc_scores)
        df = pd.DataFrame(zip(p, mean, std), columns=(p_name, "Mean ROC-AOC", "Standard Deviation"))
    else:
        df = pd.DataFrame(zip(mean, std), columns=("Mean ROC-AOC", "Standard Deviation"))

    df.to_csv(path, index=False)


# Questions start here.................................
def q1():
    X, y = _load_shuffle_data('./data/credit_card_train.csv')

    score_list = []

    auc_scores = _auc_training_process(DummyClassifier, X, y, strategy='most_frequent')
    score_list.append(auc_scores)

    print("Naive Classifier results: ")
    _print_results(auc_scores)
    _save_results('./results/Q1.csv', score_list)
    print("\n")


def q2():
    X, y = _load_shuffle_data('./data/credit_card_train.csv')

    # Random forest---------------------------------------------------------------------------------
    n_estimator_list = [100, 200, 300, 450]
    score_list = []

    print("RandomForest Results: ")
    for p in n_estimator_list:
        print("    {} is: {}".format('n_estimators', p))
        auc_scores = _auc_training_process(RandomForestClassifier, X, y,
                                           n_estimators=p,
                                           n_jobs=-1)
        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q2_RandomForest.csv', score_list, p_name='n_estimators', p=n_estimator_list)
    print("\n")

    # XGBOOST--------------------------------------------------------------------------------------------
    learning_rate_list = [0.01, 0.1, 0.25]
    score_list = []

    print("XGBoost Results: ")
    for p in learning_rate_list:
        print("    {} is: {}".format('learning_rate', p))
        auc_scores = _auc_training_process(XGBClassifier, X, y,
                                           learning_rate=p)

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q2_xgb.csv', score_list, p_name='learning_rate', p=learning_rate_list)
    print("\n")

    # SVM-----------------------------------------------------------------------------------------------
    c_list = [0.1, 1.0, 10.0]
    score_list = []

    print("SVM Results: ")
    for p in c_list:
        print("    {} is: {}".format('C', p))
        auc_scores = _svm_training_process(X, y, p)

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q2_SVM.csv', score_list, p_name='C', p=c_list)
    print("\n")

    # KNN--------------------------------------------------------------------------------------
    n_neighbors_list = [3, 5, 7, 9]
    score_list = []

    print("KNN Results: ")
    for p in n_neighbors_list:
        print("    {} is: {}".format('n_neighbors', p))
        auc_scores = _auc_training_process(KNeighborsClassifier, X, y,
                                           n_neighbors=p)

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q2_KNN.csv', score_list, p_name='n_neighbors', p=n_neighbors_list)
    print("\n")

    # Naive Bayes----------------------------------------------------------------------------------
    score_list = []

    print("Naive Bayes Results: ")
    auc_scores = _auc_training_process(GaussianNB, X, y)
    score_list.append(auc_scores)

    _print_results(auc_scores)
    _save_results('./results/Q2_NB.csv', score_list)
    print("\n")


def q3():
    X, y = _load_shuffle_data('./data/credit_card_train.csv')
    print("Number of data samples: {}".format(len(X)))
    sampler = SMOTE(random_state=0)
    X, y = sampler.fit_resample(X, y)
    print("Number of data samples after resample: {}".format(len(X)))
    # Shuffle Data
    np.random.seed(10)
    idx = np.random.permutation(range(len(X)))
    X, y = X[idx], y[idx]

    # Random forest---------------------------------------------------------------------------------
    n_estimator_list = [100, 200, 300, 450]
    score_list = []

    print("RandomForest Results: ")
    for p in n_estimator_list:
        print("    {} is: {}".format('n_estimators', p))
        auc_scores = _auc_training_process(RandomForestClassifier, X, y,
                                           n_estimators=p,
                                           n_jobs=-1)
        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q3_RandomForest.csv', score_list, p_name='n_estimators', p=n_estimator_list)
    print("\n")

    # XGBOOST--------------------------------------------------------------------------------------------
    learning_rate_list = [0.01, 0.1, 0.25]
    score_list = []

    print("XGBoost Results: ")
    for p in learning_rate_list:
        print("    {} is: {}".format('learning_rate', p))
        auc_scores = _auc_training_process(XGBClassifier, X, y,
                                           learning_rate=p)

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q3_xgb.csv', score_list, p_name='learning_rate', p=learning_rate_list)
    print("\n")

    # SVM-----------------------------------------------------------------------------------------------
    c_list = [0.1, 1.0, 10.0]
    score_list = []

    print("SVM Results: ")
    for p in c_list:
        print("    {} is: {}".format('C', p))
        auc_scores = _svm_training_process(X[:50000], y[:50000], p)  # Downsampled due to high computation cost on svm

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q3_SVM.csv', score_list, p_name='C', p=c_list)
    print("\n")

    # KNN--------------------------------------------------------------------------------------
    n_neighbors_list = [3, 5, 7, 9]
    score_list = []

    print("KNN Results: ")
    for p in n_neighbors_list:
        print("    {} is: {}".format('n_neighbors', p))
        auc_scores = _auc_training_process(KNeighborsClassifier, X, y,
                                           n_neighbors=p)

        _print_results(auc_scores)
        score_list.append(auc_scores)

    _save_results('./results/Q3_KNN.csv', score_list, p_name='n_neighbors', p=n_neighbors_list)
    print("\n")

    # Naive Bayes----------------------------------------------------------------------------------
    score_list = []

    print("Naive Bayes Results: ")
    auc_scores = _auc_training_process(GaussianNB, X, y)
    score_list.append(auc_scores)

    _print_results(auc_scores)
    _save_results('./results/Q3_NB.csv', score_list)
    print("\n")


def q4():
    X, y = _load_shuffle_data('./data/credit_card_train.csv')
    print("Number of data samples: {}".format(len(X)))
    sampler = SMOTE(random_state=0)
    X, y = sampler.fit_resample(X, y)
    print("Number of data samples after resample: {}".format(len(X)))
    # Shuffle Data
    np.random.seed(10)
    idx = np.random.permutation(range(len(X)))
    X, y = X[idx], y[idx]

    # Training start here:
    print("Using Random Forest")

    # 5 fold cv
    kf = KFold(n_splits=5)

    fold = 0
    for i_train, i_test in kf.split(X):
        print("Fold {}...... ".format(fold))
        train_X, train_y = X[i_train], y[i_train]
        test_X, test_y = X[i_test], y[i_test]

        clf = RandomForestClassifier(n_estimators=450, n_jobs=-1)
        clf.fit(train_X, train_y)

        plot_roc_curve(clf, test_X, test_y)
        plt.title("Fold {}".format(fold+1))
        plt.savefig('./results/Q4_{}.png'.format(fold))
        fold += 1


def q5():
    X, y = _load_shuffle_data('./data/credit_card_train.csv')
    print("Number of data samples: {}".format(len(X)))
    sampler = SMOTE(random_state=0)
    X, y = sampler.fit_resample(X, y)
    print("Number of data samples after resample: {}".format(len(X)))
    # Shuffle Data
    np.random.seed(10)
    idx = np.random.permutation(range(len(X)))
    X, y = X[idx], y[idx]

    test_df = pd.read_csv('./data/credit_card_test.csv')
    X_test = test_df.to_numpy()

    # Training start here:
    print("Using Random Forest")

    clf = RandomForestClassifier(n_estimators=450, n_jobs=-1)
    clf.fit(X, y)

    acc = clf.score(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    print("    Training Accuracy is: {:.4f}".format(acc))
    print("    Training ROC-AUC is: {:.4f}".format(auc))

    y_hat = clf.predict(X_test)
    np.savetxt('./results/Haizhou_Wang_labels.csv', y_hat.astype(int), fmt='%s')


if __name__ == '__main__':
    # q1()
    # q2()
    # q3()
    # q4()
    q5()

    exit()

