#!/usr/bin/python

import pandas as pd
import numpy as np

# code to gather together the code
# I am working from the CollegeSprint folder

import os

path = "/home/debasish/Desktop/CollegeSprint/Files"

file_list = sorted(os.listdir(path))

mainFrame = pd.DataFrame()
delcol = ['eob_id','batch_arrival_time','eob_plan_type','eob_patient_account_number',
		'eob_claim_type','batch_deposit_date','payer_id','claim_information_id',
		'eob_payer','payid','eob_service_line','eob_min_date_of_service_from','eob_max_date_of_service_to']

for file in file_list:
	eachFilePath = path + '/' + file
	df = pd.read_csv(eachFilePath, header=0)
	df = df.dropna(axis=0, how='all', subset=delcol)
	mainFrame = mainFrame.append(df, ignore_index=True)

mainFrame.to_csv('output.csv', index = False)




#code to clean the code further, as we above deleted only rows where all the delcol were null simultaneously.

df = pd.read_csv('output.csv', header=0)


#check no of nulls in columns still
df1 = df.isnull().sum()
df1[df1 > 0]

#output
#claim_payer_name                 4
#eob_plan_type                   13
#eob_claim_type                  13





#check the no of unique data in these columns
nullcol = df1[df1 > 0].index
for col in nullcol:
	print(col,":",len(df[col].unique()))

#output
#claim_payer_name : 11233
#eob_plan_type : 36
#eob_claim_type : 8



#delete the rows that has claim_payer_name null. We are doing this as it has many unique values
#We will impute the other two columns
df = df.dropna(axis=0, subset=['claim_payer_name'])


#we see both the columns are null together
df1 = df[(df['eob_plan_type'].isnull())]
df1[['eob_plan_type','eob_claim_type']]


#impute the 13 null entries of each column with the mode
df['eob_plan_type'] = df['eob_plan_type'].fillna(df['eob_plan_type'].mode().astype(str).iloc[0])
df['eob_claim_type'] = df['eob_claim_type'].fillna(df['eob_claim_type'].mode().astype(str).iloc[0])


df.to_csv('outputCleaned.csv', index = False)
