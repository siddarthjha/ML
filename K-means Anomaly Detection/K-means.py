# import libraries

import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Extract data
df = pd.read_csv('Sample.csv')
print(df.head())

# Understand Data
print(df.describe(include = "all"))
print(df.dtypes)
# Data types of all the column
# Data formatting


for i in range(1000):
    df['B'][i]=(df['B'][i].replace(' ', '').replace('-', '').replace(':', ''))
   
print(df.head(3))
df['B'] = df['B'].astype(float)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
print(pd.DataFrame(data_scaled).describe())


lst=['A','B','C','D','E','F']
data.columns=lst
print(data)
data.sort_values(['B', 'C'], axis=0, ascending=True, inplace=True) 
print(data.tail(50))

# Data visualization
plt.scatter(data['B'],data['C'])
plt.xlabel('B')
plt.ylabel('C')
plt.show()

data = data[['B', 'C']]
print(data)

n_cluster = range(1, 10)
print(n_cluster)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(n_cluster, scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Forming clusters
km = KMeans(n_clusters=8)
y_predicted = km.fit_predict(data[['B','C']])
print(y_predicted)
data['cluster']=y_predicted
print(data.head())
print(km.cluster_centers_)


df1 = data[data.cluster==0]
df2 = data[data.cluster==1]
df3 = data[data.cluster==2]
df4 = data[data.cluster==3]
df5 = data[data.cluster==4]
plt.scatter(df1['B'],df1['C'],color='green')
plt.scatter(df2['B'],df2['C'],color='red')
plt.scatter(df3['B'],df3['C'],color='black')
plt.scatter(df4['B'],df4['C'],color='blue')
plt.scatter(df5['B'],df5['C'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.show()

# Anomaly detection
def get_distance_by_point(data, models):
    distance = pd.Series(dtype ='float64')
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = models.cluster_centers_[models.labels_[i]-1]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance
 
outliers_fraction = 0.01

# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
data = df[['B','C']]
distance = get_distance_by_point(data, kmeans[8] )
number_of_outliers = int(outliers_fraction*len(distance))
print(number_of_outliers)
threshold = distance.nlargest(number_of_outliers).min()
print(threshold)


# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
data['anomaly'] = (distance >= threshold).astype(int)
print(df['anomaly'].head(20))
fig, ax = plt.subplots(figsize=(10,10))
colors = {0:'blue', 1:'red'}
ax.scatter(data['B'], data['C'], c=data["anomaly"].apply(lambda x: colors[x]))
plt.xlabel('Time Stamp')
plt.ylabel('System Output')
plt.show();


