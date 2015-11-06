import pandas as pd
import numpy as np
from fuzzywuzzy import process

df = pd.read_csv('outputCleaned.csv', header=0)
df1 = pd.DataFrame()
df1['claim_payer_name'] = df['claim_payer_name']
names = df1['claim_payer_name'].unique().tolist()
#df = []
matches = dict()
i = 0
count = 0
for name in names:
	i = i + 1
	if not name in matches:
		matches[name] = name
		count = count + 1
		for nm in names[i:]:
			if nm not in matches:
				temp = list()
				temp.append(nm)
				match = process.extractOne(name, temp)
				if (match[1] > 80):
					matches[str(match[0])] = name
					count = count + 1
		print (count)


print("start")
for index, row in df1.iterrows():
	if row['claim_payer_name'] in matches:
		row['claim_payer_name'] = matches[row['claim_payer_name']]

df1.to_csv('payername80.csv', index = False)

df2 = pd.DataFrame()
df2['Orginal'] = df.claim_payer_name
df2['New'] = df1.claim_payer_name
df2.to_csv('payer80.csv', index = False)
