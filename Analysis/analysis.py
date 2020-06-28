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
