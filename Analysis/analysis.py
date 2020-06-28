import pandas as pd                          # Pandas module for data analysis
import numpy as np                           # Numpy module for mathematical calculations
from matplotlib import pyplot as plt         # Pyplot module for data plotting
import seaborn as sns                        # Seaborn module for data visualization
from collections import Counter              # Counter module for count of key value pairs
# Importing all the required libraries

def Nan_percent(df, column_name):            # Function to check the % of null values present in each column 
	row_count = df[column_name].shape[0]
	empty_values = row_count - df[column_name].count()
	return (100.0 * empty_values) / row_count


d = pd.read_csv('athlete_events.csv')                  # Loads the data
print('The number of rows and columns', d.shape)
data = pd.DataFrame(d)                                 # The dataframe of the dataset is created
print(data.info())
print(data.describe(include = 'all'))
print(data.dtypes)

for i in list(d):
	print(i + ':' + str(Nan_percent(d, i)) + '%')      # Calling of Nan_percent function 

age_mean = int(data['Age'].mean())                     # Calculation of Mean 
height_mean = int(data['Height'].mean())
weight_mean = int(data['Weight'].mean())
print('The age mean which will be replaced with null values is', age_mean)
print('The height mean which will be replaced with null values is', height_mean)
print('The Weight mean which will be replaced with null values is', weight_mean)
data['Age'].replace(np.nan, age_mean, inplace = True)    # Replacing of Null values
data['Height'].replace(np.nan, height_mean, inplace = True)
data['Weight'].replace(np.nan, weight_mean, inplace = True)
# This replaces all the null values with their mean values


