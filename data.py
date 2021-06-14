import pandas as pd
import os
import numpy as np
import itertools
from tqdm import tqdm
import joblib
from geopy.distance import geodesic

tqdm.pandas()

def findYes(row):
    row = row.iloc[0]
    query = coupon_list_train_df[(coupon_list_train_df.DISPFROM <= row.I_DATE) & (coupon_list_train_df.DISPEND >= row.I_DATE)]
    query = query[query.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    query['USER_ID_hash'] = row.USER_ID_hash
    return query
    
def findNot(row):
    row = row.iloc[0]
    user_data = user_list.loc[user_list['USER_ID_hash'] == row.USER_ID_hash].iloc[0]
    query = coupon_list_train_df[(coupon_list_train_df.DISPFROM <= row.I_DATE) & (coupon_list_train_df.DISPEND >= row.I_DATE)]
    query = query[(user_data.REG_DATE <= query.DISPEND) & (query.DISPFROM <= user_data.WITHDRAW_DATE)]
    query = query[~query.COUPON_ID_hash.isin(cvdict[row.USER_ID_hash])]
    query['USER_ID_hash'] = row.USER_ID_hash
    query = query.sample(n=9, random_state=0)
    return query

print("---Load Data---")
DIR = "./data"

# Dataset
coupon_detail_train_df = pd.read_csv(os.path.join(DIR, "coupon_detail_train.csv"))
coupon_list_test_df = pd.read_csv(os.path.join(DIR, "coupon_list_test.csv"))
coupon_list_train_df = pd.read_csv(os.path.join(DIR, "coupon_list_train.csv"))
coupon_visit_train_df = pd.read_csv(os.path.join(DIR, "coupon_visit_train.csv"))
pref_loc_df = pd.read_csv(os.path.join(DIR, "prefecture_locations.csv"))
sample_sub = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
user_list = pd.read_csv(os.path.join(DIR, "user_list.csv"))

# Translator
pref = pd.read_csv(os.path.join(DIR,"pref.csv"), delimiter=';', index_col="jpn")
pref_office = pd.read_csv(os.path.join(DIR, "pref_office.csv"), delimiter=';', index_col="jpn")
small_area_name = pd.read_csv(os.path.join(DIR, "small_area_name.csv"), delimiter=';', index_col="jpn")
big_area_name = pd.read_csv(os.path.join(DIR, "big_area_name.csv"), delimiter=';', index_col="jpn")
capsule_text = pd.read_csv(os.path.join(DIR, "capsule_text.csv"), delimiter=';', index_col="jpn")
genre_name = pd.read_csv(os.path.join(DIR, "genre.csv"), delimiter=';', index_col="jpn")

print("---Translate Data---")
# CAPSULE TEXT
coupon_list_test_df.CAPSULE_TEXT = coupon_list_test_df.CAPSULE_TEXT.replace(capsule_text.to_dict()["en"])
coupon_list_train_df.CAPSULE_TEXT = coupon_list_train_df.CAPSULE_TEXT.replace(capsule_text.to_dict()["en"])

# GENRE NAME
coupon_list_test_df.GENRE_NAME = coupon_list_test_df.GENRE_NAME.replace(genre_name.to_dict()["en"])
coupon_list_train_df.GENRE_NAME = coupon_list_train_df.GENRE_NAME.replace(genre_name.to_dict()["en"])

# PREF NAME
coupon_list_test_df.ken_name = coupon_list_test_df.ken_name.replace(pref.to_dict()["en"])
coupon_list_train_df.ken_name = coupon_list_train_df.ken_name.replace(pref.to_dict()["en"])
pref_loc_df.PREF_NAME = pref_loc_df.PREF_NAME.replace(pref.to_dict()["en"])
user_list.PREF_NAME = user_list.PREF_NAME.replace(pref.to_dict()["en"])

#PREFECTUAL_OFFICE
pref_loc_df.PREFECTUAL_OFFICE = pref_loc_df.PREFECTUAL_OFFICE.replace(pref_office.to_dict()["en"])

#SMALL_AREA_NAME
coupon_detail_train_df.SMALL_AREA_NAME = coupon_detail_train_df.SMALL_AREA_NAME.replace(small_area_name.to_dict()["en"])
coupon_list_test_df.small_area_name = coupon_list_test_df.small_area_name.replace(small_area_name.to_dict()["en"])
coupon_list_train_df.small_area_name = coupon_list_train_df.small_area_name.replace(small_area_name.to_dict()["en"])

#large_area_name
coupon_list_test_df.large_area_name = coupon_list_test_df.large_area_name.replace(big_area_name.to_dict()["en"])
coupon_list_train_df.large_area_name = coupon_list_train_df.large_area_name.replace(big_area_name.to_dict()["en"])

print('---Preprocess Data---')
#cause it's annoying
coupon_visit_train_df.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)

coupon_list_train_df.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)
coupon_list_test_df.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)

coupon_list_train_df['VALIDFROM'].fillna(coupon_list_train_df['DISPFROM'], inplace=True)
coupon_list_train_df['VALIDEND'].fillna(pd.Timestamp.max, inplace=True)

coupon_list_test_df['VALIDFROM'].fillna(coupon_list_test_df['DISPFROM'], inplace=True)
coupon_list_test_df['VALIDEND'].fillna(pd.Timestamp.max, inplace=True)

coupon_list_train_df['DISPFROM'] = pd.to_datetime(coupon_list_train_df['DISPFROM'])
coupon_list_train_df['DISPEND'] = pd.to_datetime(coupon_list_train_df['DISPEND'])
coupon_list_train_df['VALIDFROM'] = pd.to_datetime(coupon_list_train_df['VALIDFROM'])
coupon_list_train_df['VALIDEND'] = pd.to_datetime(coupon_list_train_df['VALIDEND'])

coupon_list_test_df['DISPFROM'] = pd.to_datetime(coupon_list_test_df['DISPFROM'])
coupon_list_test_df['DISPEND'] = pd.to_datetime(coupon_list_test_df['DISPEND'])
coupon_list_test_df['VALIDFROM'] = pd.to_datetime(coupon_list_test_df['VALIDFROM'])
coupon_list_test_df['VALIDEND'] = pd.to_datetime(coupon_list_test_df['VALIDEND'])

coupon_list_train_df['VALIDPERIOD'].fillna((coupon_list_train_df['VALIDEND'] - coupon_list_train_df['VALIDFROM'])/np.timedelta64(1,'D'), inplace=True)
coupon_list_test_df['VALIDPERIOD'].fillna((coupon_list_test_df['VALIDEND'] - coupon_list_test_df['VALIDFROM'])/np.timedelta64(1,'D'), inplace=True)

coupon_list_train_df['VALIDPERIOD'] = coupon_list_train_df['VALIDPERIOD'].astype(int)
coupon_list_test_df['VALIDPERIOD'] = coupon_list_test_df['VALIDPERIOD'].astype(int)



coupon_list_train_df.fillna(-1, inplace=True)
coupon_list_test_df.fillna(-1, inplace=True)

user_list.WITHDRAW_DATE.fillna(pd.Timestamp.max, inplace=True)
user_list.PREF_NAME.fillna(user_list.PREF_NAME.value_counts().index[0], inplace=True)

user_list['WITHDRAW_DATE'] = pd.to_datetime(user_list['WITHDRAW_DATE'])
user_list['REG_DATE'] = pd.to_datetime(user_list['REG_DATE'])

coupon_detail_train_df = coupon_detail_train_df[['USER_ID_hash','COUPON_ID_hash','PURCHASEID_hash','I_DATE']]

print("---Save coupon_detail_train---")
print("---Save user_list---")


print("---Create Train Data---")
cvdict = coupon_detail_train_df[['USER_ID_hash','COUPON_ID_hash']].groupby('USER_ID_hash')['COUPON_ID_hash'].progress_apply(list)

user_list = pd.merge(user_list,pref_loc_df.drop('PREFECTUAL_OFFICE', axis=1),how='left')

print("---Create neg_coupon_list_train---")
neg_coupon_list_train_df = coupon_detail_train_df.groupby('PURCHASEID_hash', group_keys=False).progress_apply(findNot)
neg_coupon_list_train_df = pd.merge(neg_coupon_list_train_df,pref_loc_df.drop('PREFECTUAL_OFFICE', axis=1),how='left')
neg_coupon_list_train_df['TARGET'] = 0
neg_coupon_list_train_df = pd.merge(neg_coupon_list_train_df, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))


print("---Create pos_coupon_list_train---")
pos_coupon_list_train_df = coupon_detail_train_df.groupby('PURCHASEID_hash', group_keys=False).progress_apply(findYes)
pos_coupon_list_train_df = pd.merge(pos_coupon_list_train_df,pref_loc_df.drop('PREFECTUAL_OFFICE', axis=1),how='left')
pos_coupon_list_train_df['TARGET'] = 1
pos_coupon_list_train_df = pd.merge(pos_coupon_list_train_df, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))


print("---Combine positive & negative coupon_list_train---")
dataset = pd.concat([pos_coupon_list_train_df, neg_coupon_list_train_df]).reset_index(drop=True)
dataset = dataset[['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER', 'TARGET']]
dataset.sort_values('TARGET',inplace=True)
print("---Drop False Negative Data---")
dataset.drop_duplicates(subset=['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER'], keep='first', inplace=True)

print("---Compare the pretecture location of coupon and user in training---")
dataset['SAME_PREF'] = dataset['PREF_NAME_COUPON'] == dataset['PREF_NAME_USER']
dataset['LONGITUDE_DIST'] = np.vectorize(np.abs)(dataset.LONGITUDE_COUPON - dataset.LONGITUDE_USER)
dataset['LATITUDE_DIST'] = np.vectorize(np.abs)(dataset.LATITUDE_COUPON - dataset.LATITUDE_USER)
dataset['DIST'] = dataset.progress_apply(lambda x: geodesic((x['LATITUDE_COUPON'],x['LONGITUDE_COUPON']), (x['LATITUDE_USER'],x['LONGITUDE_USER'])).km, axis=1)

print("---Save coupon_list_train---")


dataset[['SEX_ID', 'USER_ID_hash', 'COUPON_ID_hash', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'CAPSULE_TEXT', 'GENRE_NAME', 'PREF_NAME_USER']] = dataset[['SEX_ID', 'USER_ID_hash', 'COUPON_ID_hash', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'CAPSULE_TEXT', 'GENRE_NAME', 'PREF_NAME_USER']].astype('category')
dataset[['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD', 'VALIDPERIOD', 'AGE']] = dataset[['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD', 'VALIDPERIOD', 'AGE']].astype(np.uint32)
dataset[['USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']] = dataset[['USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']].astype(np.int8)
dataset[['LATITUDE_USER', 'LONGITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_COUPON']] = dataset[['LATITUDE_USER', 'LONGITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_COUPON']].astype(np.float64)

joblib.dump(dataset, "coupon_list_train.pkl")

print("---Create Testing Data---")
# Permutation of User-Coupon Test
clist = coupon_list_test_df.COUPON_ID_hash.unique().tolist()
ulist = user_list.USER_ID_hash.unique().tolist()

relations = [r for r in itertools.product(clist, ulist)]
relations = pd.DataFrame(relations,columns=['COUPON_ID_hash','USER_ID_hash'])

coupon_list_test_df = pd.merge(coupon_list_test_df,pref_loc_df.drop('PREFECTUAL_OFFICE', axis=1),how='left')
coupon_list_test_df = pd.merge(relations,coupon_list_test_df,how='left')
coupon_list_test_df = pd.merge(coupon_list_test_df, user_list, how='left', on='USER_ID_hash', suffixes=('_COUPON', '_USER'))
coupon_list_test_df = coupon_list_test_df[['USER_ID_hash', 'COUPON_ID_hash', 'CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'LATITUDE_COUPON', 'LONGITUDE_COUPON', 'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME_USER', 'LATITUDE_USER', 'LONGITUDE_USER']]


print("---Compare the prefecture location of coupon and user in testing---")
coupon_list_test_df['SAME_PREF'] = coupon_list_test_df['PREF_NAME_COUPON'] == coupon_list_test_df['PREF_NAME_USER']
coupon_list_test_df['LONGITUDE_DIST'] = np.vectorize(np.abs)(coupon_list_test_df.LONGITUDE_COUPON - coupon_list_test_df.LONGITUDE_USER)
coupon_list_test_df['LATITUDE_DIST'] = np.vectorize(np.abs)(coupon_list_test_df.LATITUDE_COUPON - coupon_list_test_df.LATITUDE_USER)
coupon_list_test_df['DIST'] = coupon_list_test_df.progress_apply(lambda x: geodesic((x['LATITUDE_COUPON'],x['LONGITUDE_COUPON']), (x['LATITUDE_USER'],x['LONGITUDE_USER'])).km, axis=1)

print("---Save coupon_list_test--")


coupon_list_test_df[['SEX_ID', 'USER_ID_hash', 'COUPON_ID_hash', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'CAPSULE_TEXT', 'GENRE_NAME', 'PREF_NAME_USER']] = coupon_list_test_df[['SEX_ID', 'USER_ID_hash', 'COUPON_ID_hash', 'LARGE_AREA_NAME', 'PREF_NAME_COUPON', 'SMALL_AREA_NAME', 'CAPSULE_TEXT', 'GENRE_NAME', 'PREF_NAME_USER']].astype('category')
coupon_list_test_df[['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD', 'VALIDPERIOD', 'AGE']] = coupon_list_test_df[['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD', 'VALIDPERIOD', 'AGE']].astype(np.uint32)
coupon_list_test_df[['USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']] = coupon_list_test_df[['USABLE_DATE_MON', 'USABLE_DATE_TUE','USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY', 'USABLE_DATE_BEFORE_HOLIDAY']].astype(np.int8)
coupon_list_test_df[['LATITUDE_USER', 'LONGITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_COUPON']] = coupon_list_test_df[['LATITUDE_USER', 'LONGITUDE_USER', 'LATITUDE_COUPON', 'LONGITUDE_COUPON']].astype(np.float64)

joblib.dump(coupon_list_test_df, "coupon_list_test.pkl")

