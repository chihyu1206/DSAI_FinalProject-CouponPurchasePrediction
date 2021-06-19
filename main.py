import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import lightgbm as lgb

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rfc", help="Choose the ML method from RandomForest(rfc) or LightGBM(lgb) (default: rfc)")
    parser.add_argument("--output", default="submission.csv", help="Specify the path of submission result (default: submission.csv)")
    return parser.parse_args()

DIR = "./data"

# Load the data
train = joblib.load(os.path.join(DIR, "coupon_list_train.pkl")).sample(frac=1, random_state=0).reset_index(drop=True)
test =  joblib.load(os.path.join(DIR, "coupon_list_test.pkl"))
submission = pd.read_csv(os.path.join(DIR,'sample_submission.csv'))

# Preprocess the loaded data
train.drop(['DISPFROM', 'DISPEND', 'VALIDFROM', 'VALIDEND', 'REG_DATE', 'WITHDRAW_DATE'], axis=1, inplace=True)
test.drop(['DISPFROM', 'DISPEND', 'VALIDFROM', 'VALIDEND', 'REG_DATE', 'WITHDRAW_DATE'], axis=1, inplace=True)
x_train = train.drop(['USER_ID_hash', 'COUPON_ID_hash', 'TARGET'], axis=1)
y_train = train.TARGET.values.reshape(-1)
x_test = test.drop(['USER_ID_hash', 'COUPON_ID_hash'],axis=1)
categoricals = x_train.dtypes[x_train.dtypes == "category"].index.tolist()
x_train_d = pd.get_dummies(x_train, columns=categoricals)
x_test_d = pd.get_dummies(x_test, columns=categoricals)
x_train_d, x_test_d = x_train_d.align(x_test_d, join='left', axis=1)
x_train_d.fillna(0, inplace=True)
x_test_d.fillna(0, inplace=True)

parser = config()    
# Training
if parser.model == "rfc":
    clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1, n_estimators=240, class_weight="balanced_subsample")
    clf.fit(x_train_d, y_train)
else:
    clf = lgb.LGBMClassifier(objective="binary", verbose=1, n_jobs=-1, class_weight="balanced", n_estimators=300)
    clf.fit(x_train_d, y_train)
    
# Prediction
y_pred = clf.predict_proba(x_test_d)
sub = test[['USER_ID_hash','COUPON_ID_hash']]
sub['TARGET'] = y_pred[:,1]
    
grouped = sub.groupby('USER_ID_hash')
def get_top10(row):
    pred = grouped.get_group(row.USER_ID_hash).sort_values(by=['TARGET'],ascending=False)
    pred = ' '.join(map(str, pred.head(10).COUPON_ID_hash.values))
    return pred
submission['PURCHASED_COUPONS'] = submission.apply(get_top10, axis=1)
sub_name = parser.output
submission.to_csv(sub_name, index=False)
    