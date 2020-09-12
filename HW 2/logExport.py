import pandas as pd
import pickle

with open('./submissions/results.pkl', 'rb') as f:
    result_log = pickle.load(f)

result_log['RegularTree'].to_csv('./resultExport/reg_2.csv', index=None)
result_log['RandomForest'].to_csv('./resultExport/rndf_2.csv', index=None)
result_log['XGBoost'].to_csv('./resultExport/xgb_2.csv', index=None)