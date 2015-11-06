import pandas as pd
import numpy as np

df = pd.read_csv('clean.csv', header=0)

delcol = ['eob_id','eob_plan_type','eob_patient_account_number','eob_claim_type','batch_deposit_date','payer_id','claim_information_id','eob_payer','payid','eob_service_line','eob_min_date_of_service_from','eob_max_date_of_service_to']
df = df.drop(delcol, axis=1)

datecolumnformat1 = ['claim_file_arrival_time','bill_print_datefrom','batch_arrival_time']
datecolumnformat2 = ['claim_max_service_to_date','claim_min_service_from_date']

for col in datecolumnformat1:
	df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='raise')

for col in datecolumnformat2:
	df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='raise')

df['Delay'] = (df['batch_arrival_time'] - df['claim_file_arrival_time']).astype('timedelta64[D]')
df = df.drop(['batch_arrival_time'], axis=1)

df['Delay_Claim_bill'] = (df['claim_file_arrival_time'] - df['bill_print_datefrom']).astype('timedelta64[D]')
df['hospitalized'] = (df['claim_max_service_to_date'] - df['claim_min_service_from_date']).astype('timedelta64[D]')


df['claim_file_arrival_year'] = df['claim_file_arrival_time'].dt.year
df['claim_file_arrival_month'] = df['claim_file_arrival_time'].dt.month
df['bill_print_year'] = df['bill_print_datefrom'].dt.year
df['bill_print_month'] = df['bill_print_datefrom'].dt.month
df['claim_max_service_year'] = df['claim_max_service_to_date'].dt.year
df['claim_max_service_month'] = df['claim_max_service_to_date'].dt.month
df['claim_min_service_year'] = df['claim_min_service_from_date'].dt.year
df['claim_min_service_month'] = df['claim_min_service_from_date'].dt.month


df['claim_file_arrival_time'] = (df['claim_file_arrival_time'] - df['claim_file_arrival_time'].min()).astype('timedelta64[D]')
df['bill_print_datefrom'] = (df['bill_print_datefrom'] - df['bill_print_datefrom'].min()).astype('timedelta64[D]')
df['claim_max_service_to_date'] = (df['claim_max_service_to_date'] - df['claim_max_service_to_date'].min()).astype('timedelta64[D]')
df['claim_min_service_from_date'] = (df['claim_min_service_from_date'] - df['claim_min_service_from_date'].min()).astype('timedelta64[D]')


from sklearn import preprocessing
objcolumns = df.blocks['object'].columns
for col in objcolumns:
    print(col, df[col].isnull().sum())
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.to_csv('Data.csv', index = False)

from sklearn.cross_validation import train_test_split
train, test = train_test_split(df, test_size = 0.25)

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)

trainDelay = train.Delay
testDelay = test.Delay

train = train.drop(['Delay'], axis=1)
test = test.drop(['Delay'], axis=1)

from sklearn import preprocessing
for col in train.columns:
    scaler = preprocessing.StandardScaler()
    train[col] = scaler.fit_transform(train[col])
    test[col] = scaler.transform(test[col])

import xgboost as xgb
dtrain = xgb.DMatrix(train.values, label = trainDelay)
#train = []
dtrain.save_binary("dtrain.buffer")
#dtrain = xgb.DMatrix('dtrain.buffer')

params = {'objective': 'reg:linear',
          'eta' : 0.01,
          'max_depth' : 10,
          'subsample' : 0.9,
          'colsample_bytree': 0.9}

num_rounds = 3000
model = xgb.train(params, dtrain, num_rounds)
model.save_model('xgbmodel.model')
#model = xgb.Booster({'nthread':4})
#model.load_model("xgbmodel.model")

dtest = xgb.DMatrix(test.values)
dtest.save_binary("dtest.buffer")
#dtest = xgb.DMatrix('dtest.buffer')
#test = []
pred2 = model.predict(dtest)

df1 = pd.DataFrame()
df1["Orginal"] = testDelay
df1["Predicted"] = pred2
df1.to_csv('compareDelay.csv', index = False)


import matplotlib.pyplot as plt
plt.style.use("ggplot")
mapper = { 'f{0}' . format (I): v for I, v in  enumerate (train.columns)}
mapped = {mapper [k]: v for k, v in model.get_fscore().items()}
import operator
mapped = sorted(mapped.items(), key=operator.itemgetter(1))
df2 = pd.DataFrame(mapped, columns=['feature', 'fscore'])
df2['fscore'] = df2['fscore'] / df2['fscore'].sum()
df2.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')


xx = np.linspace(-10,500)
yy = xx
h0 = plt.plot(xx, yy, 'k-', label="ideal Values")
plt.scatter(df1.Orginal, df1.Predicted, c = 'y')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error
rmse =  mean_squared_error(df1.Orginal, df1.Predicted)**0.5

print("RMSE", rmse)

#at claim_file_information_id 376529 changed year from 0130 to 2013
#rmse 15.2201
