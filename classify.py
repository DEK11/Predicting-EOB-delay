import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)

train['Delay'] = train['Delay'].apply(lambda x: int(x/30))
test['Delay'] = test['Delay'].apply(lambda x: int(x/30))

trainDelay = train.Delay
testDelay = test.Delay

train = train.drop(['Delay'], axis=1)
test = test.drop(['Delay'], axis=1)

from sklearn import preprocessing
for col in train.columns:
    scaler = preprocessing.StandardScaler()
    train[col] = scaler.fit_transform(train[col])
    test[col] = scaler.transform(test[col])

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500, random_state = 2543, max_features = None, n_jobs = -1)
forest = forest.fit(train, trainDelay)
pred = forest.predict(test)
acc = forest.score(test, testDelay)

df = pd.DataFrame()
df["Orginal"] = testDelay
df["Predicted"] = pred
df.to_csv('classifyDelay.csv', index = False)

print("Accuracy", acc)
print("Total rows tested", df.count())
print("Matched", (df['Orginal'] == df['Predicted']).sum())
print("Mismatched", (df['Orginal'] != df['Predicted']).sum())
