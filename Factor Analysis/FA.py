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
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print(ev)

# Here, you can see only for 6-factors eigenvalues are greater than one. It means we need to choose only 6 factors (or unobserved variables).

# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=6, rotation = None)
fa.fit(df)
print(fa.loadings_)
# Create factor analysis object and perform factor analysis using 5 factors
fa = FactorAnalyzer()
fa.set_params(n_factors=5, rotation = None)
print(fa.loadings_)

# Get variance of each factors

print(fa.get_factor_variance())
