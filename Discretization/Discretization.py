# Import libraries

import numpy as np
import pandas as pd

np.random.seed(123)
# create 10 random integers  
x = np.random.randint(low=25, high=200, size=10)

x = np.sort(x)
print(x)
# array([ 42,  82,  91, 108, 121, 123, 131, 134, 148, 151])
# digitize examples
np.digitize(x,bins=[50])
np.digitize(x,[50,100])
np.digitize(x,[25,50,100])
df = pd.DataFrame({"height":x})
df.head()
df['binned']=pd.cut(x=df['height'], bins=[0,25,50,100,200])
df.head()
df['height_bin']=pd.cut(x = df['height'],
                        bins = [0,25,50,100,200], 
                        labels = [0, 1, 2,3])

df['height_bin']=pd.cut(x=df['height'], bins=[0,25,50,100,200], 
                        labels=["very short", " short", "medium","tall"])
print(df.head())

