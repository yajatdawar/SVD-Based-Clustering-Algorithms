# Author : Yajat Dawar
# Implementation of SVD Based Spectral Clustering on sample data points.

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from scipy.linalg import svd
from scipy import spatial
import sklearn.metrics as metrics

np.set_printoptions(suppress=True)


def similarity(a,b):
    return np.exp(-1.0*spatial.distance.euclidean(a,b)/2*0.6*0.6) # taking sigma = 0.6

data = make_blobs(n_samples=200, n_features=2,centers = 3, cluster_std=1.6, random_state=50)

# Uncomment the following to see data points
# print(data[0])

sim_matrix = [[similarity(data[0][i],data[0][j]) for j in range(len(data[0]))] for i in range(len(data[0]))]

# Uncomment the following to see the similarity matrix
#print(sim_matrix)

U, s, VT = svd(sim_matrix)
# Taking L = 3;The first 3 coloumns of U matrix
Temp = U[:,[0,1,2]]

# Uncomment the following to see the Truncated U matrix of SVD output
#print(Temp)

from sklearn.cluster import KMeans

# creating the kmeans object

kmeans = KMeans(n_clusters=3)

# fitting kmeans object to data

kmeans.fit(Temp)

#print location of clusters learned by kmeans object
#print(kmeans.cluster_centers_)

# create scatter plot to print the data

y_km = kmeans.fit_predict(Temp)
#print(y_km)


cluster1 = [data[0][i] for i in range(200) if(y_km[i]==0)]
cluster2 = [data[0][i] for i in range(200) if(y_km[i]==1)]
cluster3 = [data[0][i] for i in range(200) if(y_km[i]==2)]

cluster1X = [cluster1[i][0] for i in range(len(cluster1))]
cluster1Y = [cluster1[i][1] for i in range(len(cluster1))]

cluster2X = [cluster2[i][0] for i in range(len(cluster2))]
cluster2Y = [cluster2[i][1] for i in range(len(cluster2))]

cluster3X = [cluster3[i][0] for i in range(len(cluster3))]
cluster3Y = [cluster3[i][1] for i in range(len(cluster3))]

cluster3X = [cluster3[i][0] for i in range(len(cluster3))]
cluster3Y = [cluster3[i][1] for i in range(len(cluster3))]


plt.scatter(cluster1X, cluster1Y, s=100, c='red')
plt.scatter(cluster2X, cluster2Y, s=100, c='black')
plt.scatter(cluster3X, cluster3Y, s=100, c='cyan')


