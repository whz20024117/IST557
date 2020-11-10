import pandas as pd
import swifter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Define pandas apply functions..
def wordInTitle(series):
    return sum([int(series['product_title'].find(w)>=0) for w in series['search_term'].split()])
def wordInDescription(series):
    return sum([int(series['product_description'].find(w)>=0) for w in series['search_term'].split()])
def wordInAttr(series):
    return sum([int(series['attr'].find(w)>=0) for w in series['search_term'].split()])


def normAvgWordCountTitle(series):
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        count += series['product_title'].count(w)
    count /= len(series['search_term'].split())  # average

    return count / len(series['product_title'].split())  # Normalized by text length


def normAvgWordCountDescription(series):
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        count += series['product_description'].count(w)
    count /= len(series['search_term'].split())  # average

    return count / len(series['product_description'].split()) * 10  # Normalized by text length. *10 to scale up


def normAvgWordCountAttr(series):
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        count += series['product_description'].count(w)
    count /= len(series['search_term'].split())  # average

    return count / len(series['product_description'].split()) * 10  # Normalized by text length. *10 to scale up


def phraseCountTitle(series):
    pass
def phraseCountDescription(series):
    pass
def phraseCountAttr(series):
    pass


train_all = pd.read_csv('./train_all.csv')
test_all = pd.read_csv('./test_all.csv')

# Create data for training/validation/testing
X_df = pd.DataFrame(train_all[['id']])
X_df['title_score'] = train_all.swifter.apply(normAvgWordCountTitle, axis=1)
X_df['description_score'] = train_all.swifter.apply(normAvgWordCountDescription, axis=1)
X_df['attr_score'] = train_all.swifter.apply(normAvgWordCountAttr, axis=1)

y_df = pd.DataFrame(train_all[['id', 'relevance']])

test_X_df = pd.DataFrame(test_all[['id']])
test_X_df['title_score'] = test_all.swifter.apply(normAvgWordCountTitle, axis=1)
test_X_df['description_score'] = test_all.swifter.apply(normAvgWordCountDescription, axis=1)
test_X_df['attr_score'] = test_all.swifter.apply(normAvgWordCountAttr, axis=1)

# ###################### Start with model evaluation for tuning ###############################3
# print("\nStart training for tuning....")
# X = X_df[['title_score', 'description_score', 'attr_score']].to_numpy()
# y = y_df[['relevance']].to_numpy()
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)
#
# reg = XGBRegressor(learning_rate=0.25,
#                    eval_metric='rmse',
#                    max_depth=2,
#                    reg_lambda=100.0,
#                    n_estimators=150,
#                    verbosity=0)
# reg.fit(X_train, y_train)
#
# # Test
# print("Testing....")
# pred = reg.predict(X_val)
# err = mean_squared_error(y_val, pred, squared=False)


########################## LR ####################
# from sklearn.linear_model import LinearRegression
# print("\nStart LR")
# X = X_df[['title_score', 'description_score', 'attr_score']].to_numpy()
# y = y_df[['relevance']].to_numpy()
#
# X_test = test_X_df[['title_score', 'description_score', 'attr_score']].to_numpy()
#
# reg = LinearRegression()
# reg.fit(X, y)
#
# pred = reg.predict(X_test).flatten().clip(0,3)


########################## Submission ####################
print("\nStart Submission")
X = X_df[['title_score', 'description_score', 'attr_score']].to_numpy()
y = y_df[['relevance']].to_numpy()

X_test = test_X_df[['title_score', 'description_score', 'attr_score']].to_numpy()

reg = XGBRegressor(learning_rate=0.25,
                   eval_metric='rmse',
                   max_depth=2,
                   reg_lambda=100.0,
                   n_estimators=150,
                   verbosity=0)
reg.fit(X, y)

pred = reg.predict(X_test).flatten().clip(0,3)

sample_submission = pd.read_csv('data/sample_submission.csv.zip')
submission = pd.DataFrame(columns=sample_submission.columns)

submission['id'] = test_X_df['id']
submission['relevance'] = pred

submission.to_csv('submission.csv', index=False)
