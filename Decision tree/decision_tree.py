import pandas as pd
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

#Load the dataset

AH_data = pd.read_csv("Decision tree.csv")

data_clean = AH_data.dropna()
print('Hello')

data_clean.dtypes
data_clean.describe()

