# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("bfi.csv")

print(df.columns)

df.drop(['gender', 'education', 'age'], axis=1, inplace=True)

df.dropna(inplace=True)
print(df.info())
print(df.head())
# Before you perform factor analysis, you need to evaluate the “factorability” of our dataset.

chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)


# Create factor analysis object and perform factor analysis

fa = FactorAnalyzer()
fa.set_params(n_factors=25, rotation = None)
fa.fit(df)
