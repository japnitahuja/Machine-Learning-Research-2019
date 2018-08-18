import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
#raw data has the dataset name too
raw_data = open("values.txt").read().split("\n")
raw_data = [list(map(str, i.split(","))) for i in raw_data]
dataset_names = [i[3] for i in raw_data]
for i in range(len(raw_data)):
    raw_data[i].pop()
raw_data = np.array(raw_data).astype('float') 
kmeans = KMeans(n_clusters = 8, random_state=0).fit(raw_data) #what's random seed

labels = kmeans.labels_

clusters = 8
data = [[] for i in range(0,clusters)]

for i in range(len(raw_data)):
    data[labels[i]].append(raw_data[i])

for i in range(len(raw_data)):
    print(dataset_names[i] + " " * (20 - len(dataset_names[i])) + str(labels[i]))
print(labels)

"""
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
"""


    
    
