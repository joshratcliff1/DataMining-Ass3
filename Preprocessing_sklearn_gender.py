import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import datetime

# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("8410_data.csv")

# Delete unnessecary atttributes
df.drop(['Mode','Q3','Q8a','Q8b','Q8c','Q8d','Q8e','Q8f','Q8g','Q8h','module'], axis=1, inplace=True)
df.drop(['Q11','Q11_CLICK','Q11a','Q11b','Q12','Q13a','Q13b','Q13c','Q13d','Q13e','Q13f'], axis=1, inplace=True)
df.drop(['Q14','Q15','Q16','Q17a','Q17b','Q17c','Q17d','Q17e','Q17f','Q18'], axis=1, inplace=True)
df.drop(['Q19a','Q19b','Q19c','Q19d','Q19e','Q19f','Q19g'], axis=1, inplace=True)
df.drop(['Sect_0_time','Sect_1_time','Sect_2_time','Sect_3_time','Sect_4_time','Sect_5_time'], axis=1, inplace=True)
df.drop(['Sect_6_time','Sect_7_time','Sect_8_time','Sect_9_time','Sect_10_time','Sect_11_time'], axis=1, inplace=True)
df.drop(['Sect_12_time','Sect_13_time','Sect_14_time','Sect_15_time','Sect_16_time'], axis=1, inplace=True)
# df.drop(['p_state','p_education_sdc','p_age_group_sdc'], axis=1, inplace=True)

# Drop the 42 rows missing an age_group and 4 rows missing a gender
df.dropna(how='any', subset=['p_age_group_sdc'], inplace=True)
df.dropna(how='any', subset=['p_gender_sdc'], inplace=True)
df.dropna(how='any', subset=['p_education_sdc'], inplace=True)

# Rename the columns
df = df.rename(columns={'p_gender_sdc': 'gender','p_age_group_sdc': 'age_group','p_education_sdc': 'education'})

# Convert columns to numeric
df.replace({'"': ''}, regex=True, inplace=True)
cols = ['Q1', 'Q2', 'Q4', 'Q6a', 'Q6b', 'Q6c', 'Q6d', 'Q7a', 'Q7b', 'Q7c', 'Q7d', 'Q7e',
        'Q10a', 'Q10b', 'Q10c', 'Q10d', 'sDevType','sOSName', 'gender']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)



'''Final Checks - Printing Values'''
print(df)
print(df.isnull().sum(axis = 0))
# print(df.Q4.describe())
print(df.dtypes)
# print(df['age_gender'].value_counts())


# Create a csv for the merged datasets
df.to_csv('8410_processed_sklearn_FULL.csv', index=False)