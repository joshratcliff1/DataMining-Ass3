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
df.drop(['srcid','Mode','Q3','Q8a','Q8b','Q8c','Q8d','Q8e','Q8f','Q8g','Q8h','module'], axis=1, inplace=True)
df.drop(['Q11','Q11_CLICK','Q11a','Q11b','Q12','Q13a','Q13b','Q13c','Q13d','Q13e','Q13f'], axis=1, inplace=True)
df.drop(['Q14','Q15','Q16','Q17a','Q17b','Q17c','Q17d','Q17e','Q17f','Q18'], axis=1, inplace=True)
df.drop(['Q19a','Q19b','Q19c','Q19d','Q19e','Q19f','Q19g'], axis=1, inplace=True)
df.drop(['Sect_0_time','Sect_1_time','Sect_2_time','Sect_3_time','Sect_4_time','Sect_5_time'], axis=1, inplace=True)
df.drop(['Sect_6_time','Sect_7_time','Sect_8_time','Sect_9_time','Sect_10_time','Sect_11_time'], axis=1, inplace=True)
df.drop(['Sect_12_time','Sect_13_time','Sect_14_time','Sect_15_time','Sect_16_time'], axis=1, inplace=True)
df.drop(['p_state','p_education_sdc'], axis=1, inplace=True)

# Drop the 42 rows missing an age_group and 4 rows missing a gender
df.dropna(how='any', subset=['p_age_group_sdc'], inplace=True)
df.dropna(how='any', subset=['p_gender_sdc'], inplace=True)

# Create new target attribute
df['age_gender'] = df['p_age_group_sdc'] + df['p_gender_sdc']

# Update age_gender
def age_gender_update(row,attribute):
    if row['age_gender'] == '"1""1"':
        return "18-29 Male"
    elif row['age_gender'] == '"1""2"':
        return "18-29 Female"
    elif row['age_gender'] == '"2""1"':
        return "30-49 Male"
    elif row['age_gender'] == '"2""2"':
        return "30-49 Female"
    elif row['age_gender'] == '"3""1"':
        return "50-64 Male"
    elif row['age_gender'] == '"3""2"':
        return "50-64 Female"
    elif row['age_gender'] == '"4""1"':
        return "65+ Male"
    elif row['age_gender'] == '"4""2"':
        return "65+ Female"
    else:
        return 'Error'

attribute = "age_gender"
df[attribute] = df.apply(lambda row: age_gender_update(row,attribute), axis=1)

# Drop the age and gender attributes after they have been combined
df.drop(['p_age_group_sdc','p_gender_sdc'], axis=1, inplace=True)



'''Final Checks - Printing Values'''
print(df)
print(df.isnull().sum(axis = 0))
# print(df.Q4.describe())
print(df['age_gender'].value_counts())


# Create a csv for the merged datasets
df.to_csv('8410_processed.csv', index=False)