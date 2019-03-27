import csv

labels = open("dt_vs_knn.txt").read().split("\n")

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

nan_measures = []
cnt_nan = 0
for cluster in range(100):
    for j in measures.keys():
        if measures[j][cluster] == float("NaN"):
            cnt_nan += 1
            if j not in nan_measures:
                nan_measures.append(j)
print("nan_measures", nan_measures)
print("baseline", cnt_nan)

inf_measures = []
cnt_inf = 0
for cluster in range(100):
    for j in measures.keys():
        if measures[j][cluster] == float("Inf"):
            cnt_inf += 1
            if j not in inf_measures:
                inf_measures.append(j)
print("ing_measures", inf_measures)
print("baseline inf", cnt_inf)



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

y = [None for i in range(30000)]
#labels
for i in labels:
    if len(i) > 1:
        i = i.split(",")
        index = int(i[0])

        if index >= 400 and index < 600:
            continue

        i = int(i[1])
        y[index] = int(i)
'''
cnt_labels = [0,0,0]
cnt_all = 0
for i in range(1000):
    if y[i] != None:
        cnt_all += 1
        cnt_labels[y[i]] += 1

for i in range(3):
    cnt_labels[i] /= cnt_all
    #cnt_labels[i] *= 100

print("class distribution:", cnt_labels)
print("cnt missing:", cnt_missing)
'''


nan_measures = []
for cluster in range(400):
    for j in measures.keys():
        if measures[j][cluster] == float("NaN"):
            cnt_nan += 1
            if j not in nan_measures:
                nan_measures.append(j)
print("nan_measures", nan_measures)
print("cluster wise", cnt_nan)

inf_measures = []
cnt_inf = 0
for cluster in range(400):
    for j in measures.keys():
        if measures[j][cluster] == float("Inf"):
            cnt_inf += 1
            if j not in inf_measures:
                inf_measures.append(j)

print("inf_measures",inf_measures)
print("cluster inf", cnt_inf)



