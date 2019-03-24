import numpy as np 
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
debug = 0


#format the baseline complexity based meta features
measures = {"overlapping.F1":[0 for i in range(1000)],
            "overlapping.F1v":[0 for i in range(1000)],
            "overlapping.F2":[0 for i in range(1000)],
            "overlapping.F3": [0 for i in range(1000)],
            "overlapping.F4":[0 for i in range(1000)],
            "neighborhood.N1":[0 for i in range(1000)],
            "neighborhood.N2":[0 for i in range(1000)],
            "neighborhood.N3":[0 for i in range(1000)],
            "neighborhood.N4":[0 for i in range(1000)],
            "neighborhood.T1":[0 for i in range(1000)],
            "neighborhood.LSCAvg":[0 for i in range(1000)],
            "linearity.L1":[0 for i in range(1000)],
            "linearity.L2":[0 for i in range(1000)],
            "linearity.L3":[0 for i in range(1000)],
            "dimensionality.T2":[0 for i in range(1000)],
            "dimensionality.T3":[0 for i in range(1000)],
            "dimensionality.T4":[0 for i in range(1000)],
            "balance.C1":[0 for i in range(1000)],
            "balance.C2":[0 for i in range(1000)],
            "network.Density":[0 for i in range(1000)],
            "network.ClsCoef":[0 for i in range(1000)],
            "network.Hubs":[0 for i in range(1000)]}

file = open("complexity_baseline.txt").read().split("\n")
count = 0

vis_baseline = [0 for i in range(400)]

for i in file:
    if i =="":
        continue
    if i =='"x"':
        continue
    if count == 23:
        count = 0
    i = i.split(",")
    if count == 0:
        temp = i[1].split("\"")
        index = int(temp[1].strip(".txt"))
        vis_baseline[index] = 1
        count+=1
    else:
        temp = i[0].split("\"")
        value = float(i[1])
        measure = measures[temp[1]]
        measure[index] = value
        count += 1

baseline_complexity = [[None for i in range(22)] for j in range(100)]
for i in range(100):
    cnt = 0
    for j in measures.keys():
        #print(i, j, result[i][cnt], len(measures[j]))
        baseline_complexity[i][cnt] = measures[j][i]
        cnt += 1

#format the cluster wise complexity based meta features
measures = {"overlapping.F1":[0 for i in range(1000)],
            "overlapping.F1v":[0 for i in range(1000)],
            "overlapping.F2":[0 for i in range(1000)],
            "overlapping.F3": [0 for i in range(1000)],
            "overlapping.F4":[0 for i in range(1000)],
            "neighborhood.N1":[0 for i in range(1000)],
            "neighborhood.N2":[0 for i in range(1000)],
            "neighborhood.N3":[0 for i in range(1000)],
            "neighborhood.N4":[0 for i in range(1000)],
            "neighborhood.T1":[0 for i in range(1000)],
            "neighborhood.LSCAvg":[0 for i in range(1000)],
            "linearity.L1":[0 for i in range(1000)],
            "linearity.L2":[0 for i in range(1000)],
            "linearity.L3":[0 for i in range(1000)],
            "dimensionality.T2":[0 for i in range(1000)],
            "dimensionality.T3":[0 for i in range(1000)],
            "dimensionality.T4":[0 for i in range(1000)],
            "balance.C1":[0 for i in range(1000)],
            "balance.C2":[0 for i in range(1000)],
            "network.Density":[0 for i in range(1000)],
            "network.ClsCoef":[0 for i in range(1000)],
            "network.Hubs":[0 for i in range(1000)]}

file = open("complexity_cluster.txt").read().split("\n")
count = 0

vis_cluster = [0 for i in range(400)]

for i in file:
    if i =="":
        continue
    if i =='"x"':
        continue
    if count == 23:
        count = 0
    i = i.split(",")
    if count == 0:
        temp = i[1].split("\"")
        index = int(temp[1].strip(".txt"))
        vis_cluster[index] = 1
        count+=1
    else:
        temp = i[0].split("\"")
        value = float(i[1])
        measure = measures[temp[1]]
        measure[index] = value
        count += 1

cluster_wise_complexity = [[None for i in range(22)] for j in range(400)]
for i in range(400):
    cnt = 0
    for j in measures.keys():
        #print(i, j, result[i][cnt], len(measures[j]))
        cluster_wise_complexity[i][cnt] = measures[j][i]
        cnt += 1

cluster_wise_complexity = np.array(cluster_wise_complexity).astype(np.float64)
baseline_complexity = np.array(baseline_complexity).astype(np.float64)

with open("cluster_wise_accuracy.txt", "r") as f:
    reader = csv.reader(f)
    cluster_wise_accuracy = [row for row in reader]
    cluster_wise_accuracy = np.array(cluster_wise_accuracy).astype(np.float64)


with open("hybrid_overall_acc.txt", "r") as f:
    reader = csv.reader(f)
    line = [row for row in reader]

baseline_accuracy = line[2][:]
baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)


#calculate difference between baseline complexity and cluster wise complexity
for feature in range(22):
    for n0 in range(100):
        for cluster in range(4):
            cluster_wise_complexity[n0*4+cluster][feature] -= baseline_complexity[n0][feature]

#calculate difference between baseline accuracy and cluster wise accuracy
for n0 in range(100):
    for cluster in range(4):
        cluster_wise_accuracy[n0*4+cluster][3] -= baseline_accuracy[n0]

plt.figure(figsize=(30,30))
for feature in range(22):
    print("feature", feature)
    x = []
    y = []
    for i in range(400):
        if vis_cluster[i] == 1 and vis_baseline[i//4] == 1:
            #print(i, i//4,vis_baseline[i//4])
            x.append(cluster_wise_complexity[i][feature])
            y.append(cluster_wise_accuracy[i][3])
    #print(len(x))
    #print(len(y))
    #plt.clf()
    plt.subplot(5,5,feature+1)
    plt.scatter(x, y)
    #fig = plt.figure()
    plot_title = "complexity " + str(feature)
    plt.title(plot_title)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
#plt.show()

#os.chdir(path)
plt_name = str(feature) + ".png"
plt.savefig(plt_name, dpi=100)
#os.chdir(home)
