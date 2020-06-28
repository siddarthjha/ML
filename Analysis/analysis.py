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

total = data.shape[0]                                    # Calculates total number of rows
athlete = len(data.ID.unique())                          # Calculate number of participants 
medal_win = len(data[data.Medal.fillna('None')!='None'].Name.unique()) # Calulates the participants won medals
print('Total number of athletes are ', athlete)
print("The number of athletes won the medals", medal_win)

print(data[data.Medal.fillna('None') != 'None'].Medal.value_counts()) # Calulates the total medals ditribution
print('The total number of medals ', data[data.Medal.fillna('None') != 'None'].shape[0]) # Total medals
print('The cities are \n', data[data.City.fillna('None') !='None'].City.value_counts()) # Retreives all the cities 
print('The number of male and female athletes\n', data[data.Sex.fillna('None') != 'None'].Sex.value_counts())
# Calculates the male and female athletes
print(data.groupby(['Team','Medal']).Medal.agg('count')) # Calulates the medals won by each team
print(data.groupby(['Sex','Medal']).Sex.agg('count'))    # Calulates the medals won by Sex
print('The different types of sport for the athletes and their participation\n', data[data.Sport.fillna('None') != 'None'].Sport.value_counts())
# Calculates all the sports and their participation 
print('Total number of women participants', len(data[data.Sex=='F'].Name.unique())) # Women participation
print('Total number of men participants', len(data[data.Sex=='M'].Name.unique()))   # Men participation
f_year_count = data[data.Sex=='F'].groupby('Year').agg('count').Name
m_year_count = data[data.Sex=='M'].groupby('Year').agg('count').Name
plot = (sns.scatterplot(data = m_year_count), sns.scatterplot(data = f_year_count))
plt.show() # Data visulation plot of male and female participants


