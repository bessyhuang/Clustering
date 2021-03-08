from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd



X = np.array([[1, 3], [3, 3], [8, 3],[7, 4], [5, 9], [10, 1]])

clustering = DBSCAN(eps=2, min_samples=2)
clustering.fit(X)

Y_pred = clustering.labels_
print(Y_pred)

# display by groups
frame = pd.DataFrame(X)
frame['Cluster'] = Y_pred
print(frame)
#print(frame['Cluster'].value_counts())

print('\n======== 每群個數 ========\n', frame.groupby('Cluster').agg({'Cluster':'count'}))

sectors = frame.groupby('Cluster')
sectors_len = len(sectors)
#print(frame['Cluster'], sectors_len)

for clusterN in range(-1, sectors_len -1):
    print('\n========= Cluster {} =========='.format(clusterN))
    print(sectors.get_group(clusterN))
