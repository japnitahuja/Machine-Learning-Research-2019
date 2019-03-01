#format the complexity based meta features

import csv

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

cluster_wise_acc = [[None for i in range(4)] for i in range(100)]
#read cluster_wise_accuracy
with open("cluster_wise_accuracy.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        cluster_wise_acc[int(row[0])*10+int(row[1])][int(row[2])] = float(row[3])

file = open("complexity.txt").read().split("\n")
count = 0

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
        count+=1
    else:
        temp = i[0].split("\"")
        value = float(i[1])
        measure = measures[temp[1]]
        measure[index] = value
        count += 1

for measure_i in measures.keys():
    acc_wise = [[] for i in range(4)]
    file_name = str(measure_i) + ".txt"
    #file = open(file_name ,"w")
    count = 0
    temp = []
    for j in measures[measure_i]:
        if len(temp) == 4:
            '''
            with open(file_name, "a") as f:
                writer = csv.writer(f)
                writer.writerow(temp)
            '''
            #sort the measures according to decressing accuracy
            vis = [0 for i in range(4)]
            temp_measure = []
            for i in range(4):
                index = None
                temp_acc = 0
                n0 = count//4 - 1
                #print(n0)
                for j in range(4):
                    if float(cluster_wise_acc[n0][j]) >= temp_acc and vis[j] == 0:
                        index = j
                        temp_acc = float(cluster_wise_acc[n0][j])
                #print(i, index)
                vis[index] = 1
                temp_measure.append(temp[index])
            #print(cluster_wise_acc[count//4])
            #print(temp)
            #print(temp_measure)
            for k in range(4):
                acc_wise[k].append(temp_measure[k])
            temp = []

        '''
        if count == 400:
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerows(acc_wise)
            break
        '''
        if count == 400:
            with open("complexity_list.csv", "a") as f:
                writer = csv.writer(f)
                temp_list = [[] for i in range(4)]
                temp_list[0].append(measure_i)
                writer.writerows(temp_list)

            with open("complexity_format.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(acc_wise)
                break

        temp.append(j)
        #print(temp)
        count +=1

