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


def normAvgWordCountTitle(series)->float:
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        count += series['product_title'].count(w)
    count /= len(series['search_term'].split())  # average

    return count / len(series['product_title'].split())  # Normalized by text length


def normAvgWordCountDescription(series)->float:
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        count += series['product_description'].count(w)
    count /= len(series['search_term'].split())  # average

    return count / len(series['product_description'].split()) * 10  # Normalized by text length. *10 to scale up


def normAvgWordCountAttr(series)->float:
    # Average count of single word matching
    count = 0
    for w in series['search_term'].split():
        try:
            count += series['attr'].count(w)
        except AttributeError:
            return 0.0
    count /= len(series['search_term'].split())  # average

    return count / len(series['attr'].split()) * 10  # Normalized by text length. *10 to scale up


def phraseCountTitle(series):
    pass
def phraseCountDescription(series):
    pass
def phraseCountAttr(series):
    pass

# searchTerm, title, description length
def getTermLength(series)->int:
    return len(series['search_term'].split())

def getTitleLength(series)->int:
    return len(series['product_title'].split())

def getDescriptionLength(series)->int:
    return len(series['product_description'].split())


train_all = pd.read_csv('./train_all.csv')
test_all = pd.read_csv('./test_all.csv')

train_dist = pd.read_csv('./train_dist.csv')
test_dist = pd.read_csv('./test_dist.csv')

# Create data for training/validation/testing
# Training
X_df = pd.DataFrame(train_all[['id']])

X_df['title_score'] = train_all.swifter.apply(normAvgWordCountTitle, axis=1)
X_df['description_score'] = train_all.swifter.apply(normAvgWordCountDescription, axis=1)
X_df['attr_score'] = train_all.swifter.apply(normAvgWordCountAttr, axis=1)
X_df['term_len'] = train_all.swifter.apply(getTermLength, axis=1)
X_df['title_len'] = train_all.swifter.apply(getTitleLength, axis=1)
X_df['description_len'] = train_all.swifter.apply(getDescriptionLength, axis=1)

X_df = X_df.merge(train_dist, how='left', on='id')


# Testing
y_df = pd.DataFrame(train_all[['id', 'relevance']])

test_X_df = pd.DataFrame(test_all[['id']])
test_X_df['title_score'] = test_all.swifter.apply(normAvgWordCountTitle, axis=1)
test_X_df['description_score'] = test_all.swifter.apply(normAvgWordCountDescription, axis=1)
test_X_df['attr_score'] = test_all.swifter.apply(normAvgWordCountAttr, axis=1)
test_X_df['term_len'] = test_all.swifter.apply(getTermLength, axis=1)
test_X_df['title_len'] = test_all.swifter.apply(getTitleLength, axis=1)
test_X_df['description_len'] = test_all.swifter.apply(getDescriptionLength, axis=1)

test_X_df = test_X_df.merge(test_dist, how='left', on='id')

features = ['title_score', 'description_score', 'attr_score', 
            'term_len', 'title_len', 'description_len', 'title_dist']


###################### Start with model evaluation for tuning ###############################3
# print("\nStart training for tuning....")
# X = X_df[features].to_numpy()
# y = y_df[['relevance']].to_numpy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

# reg = XGBRegressor(learning_rate=0.25,
#                    eval_metric='rmse',
#                    max_depth=2,
#                    reg_lambda=100.0,
#                    n_estimators=150,
#                    verbosity=0)
# reg.fit(X_train, y_train)

# # Test
# pred = reg.predict(X_val)
# err = mean_squared_error(y_val, pred, squared=False)
# print(err)


########################## Submission ####################
print("\nStart Submission")
X = X_df[features].to_numpy()
y = y_df[['relevance']].to_numpy()

X_test = test_X_df[features].to_numpy()

reg = XGBRegressor(learning_rate=0.25,
                   eval_metric='rmse',
                   max_depth=3,
                   reg_lambda=300.0,
                   n_estimators=150,
                   verbosity=0)
# reg = XGBRegressor(learning_rate=0.25,
#                    eval_metric='rmse',
#                    max_depth=2,
#                    reg_lambda=100.0,
#                    n_estimators=150,
#                    verbosity=0)
reg.fit(X, y)

pred = reg.predict(X_test).flatten().clip(1,3)

sample_submission = pd.read_csv('data/sample_submission.csv.zip')
submission = pd.DataFrame(columns=sample_submission.columns)

submission['id'] = test_X_df['id']
submission['relevance'] = pred

submission.to_csv('submission_ckpt2.csv', index=False)
