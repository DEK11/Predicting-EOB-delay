import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)

delcol = ['claim_file_arrival_year','claim_file_arrival_month','bill_print_year','bill_print_month','claim_min_service_year','claim_max_service_year','claim_frequency_type_code','claim_min_service_month','claim_payer_name']

train = train.drop(delcol, axis=1)
test = test.drop(delcol, axis=1)

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

df2 = pd.DataFrame()
df2["Orginal"] = testDelay
df2["Predicted"] = pred2
df2.to_csv('compareDelay.csv', index = False)


import matplotlib.pyplot as plt
plt.style.use("ggplot")
mapper = { 'f{0}' . format (I): v for I, v in  enumerate (train.columns)}
mapped = {mapper [k]: v for k, v in model.get_fscore().items()}
import operator
mapped = sorted(mapped.items(), key=operator.itemgetter(1))
xgb.plot_importance(model)
plt.show()
df = pd.DataFrame(mapped, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')


xx = np.linspace(-10,500)
yy = xx
h0 = plt.plot(xx, yy, 'k-', label="ideal Values")
plt.scatter(df2.Orginal, df2.Predicted, c = 'y')
plt.legend()
plt.show()
