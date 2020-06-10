import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline 


os.chdir(r"D:\Intern\fake-and-real-news-dataset") # It changes the directory to search for the files included in program

d = pd.read_csv('Fake.csv')
data = pd.DataFrame(d)

print('The top two rows of the dataset:')
print(data.head(2))
print('The bottom two rows of the dataset:')
print(data.tail(2))

#print('We will add our manual headers to validate the understanding')

#list=['First', 'Second', 'Third', 'Fourth']
#data.columns = list
data.dropna(subset =['title'], axis = 0)       # This will not change the data frame.If you need to change then inplace = True must be added
# This function remove all the rows which have no value in title column

print(data.tail(5))
# To save the dataframe in csv format use to_csv() method---> data.to_csv('Modified_data.csv')

# The main data types stored in Pandas dataframes are object, float, int, bool and datetime64. In order to better learn about each attribute, 
# it is always good for us to know the data type of each column.

print(data.dtypes) # It gives you the information of all the columns with respective data types
print(data.describe()) # It gives the statistical summary of the data(numeric data types)
print(data.describe(include='all')) # It gives you the summary of object data type included
print(data[['title']].describe())   # It gives the statistical summary of particular column
print(data.info())   # It gives you concise information of your dataset
x = data.isnull()
print(x.head())

"""

df.replace("np.nan", " x", inplace = True)  To replace the null values with "x"
df.to_csv('x.csv') Saves the file 
df=df['title'].astype("float") Converts the data type of given column
pd.get_dummies(df['column name']) Create the dummies column with the values presenrt in that column  
df.rename(columns = {'text':'text1', 'title': 'Title1'}, inplace = True) Renames the column name
df.drop('column name', axis = 1, inplace = True) Drops the original column from dataframe
df['Column name'].value_counts() It gives you the all the categories and number of rows corresponding to particular category

sns.regplot(x="engine-size", y="price", data=df) To visualize the variables to understand the linear relationship
sns.boxplot(x = 'title', y = 'subject', data= data) To visualize the categorical variables with x and y of numeric dtype
df.to_frame() Converts into the dataframe
df.index.name ="New Name" It ranmes the index column name 

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'] Correlation and Causation
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pd.concat([city_data, further_city_data], sort=False) This combines two data frames into one single dataframe
"""

x = data['text']
y = data['title']
z = plt.scatter(x, y)
print(z)
print(plt.scatter(x, y))
print('The correlation of the dataframe:')
print(data.corr())  # calculate the correlation between variables of type "int64" or "float64" 
print(data['subject'].unique())
print(data.groupby(data['subject']))

