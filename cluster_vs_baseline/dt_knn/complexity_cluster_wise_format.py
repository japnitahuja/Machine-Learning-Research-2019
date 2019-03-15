#format the complexity based meta features
debug = 0
import csv

p = [[0,1,2,3],
	[0,1,3,2],
	[0,2,1,3],
	[0,2,3,1],
	[0,3,1,2],
	[0,3,2,1],
	[1,0,2,3],
	[1,0,3,2],
	[1,2,0,3],
	[1,2,3,0],
	[1,3,0,2],
	[1,3,2,0],
	[2,0,1,3],
	[2,0,3,1],
	[2,1,0,3],
	[2,1,3,0],
	[2,3,0,1],
	[2,3,1,0],
	[3,0,1,2],
	[3,0,2,1],
	[3,1,0,2],
	[3,1,2,0],
	[3,2,0,1],
	[3,2,1,0]]

def match(X_centroids):
	dis = [0 for i in range(len(p))]
	dis_min = None
	for i in range(len(p)):
		for j in range(4):
			for feature in range(11):
				dis[i] += abs(float(centroids[0][j][feature]) - float(X_centroids[p[i][j]][feature]))
		if i == 0:
			dis_min = i
		else:
			if dis[i] < dis[dis_min]:
				dis_min = i
	if debug == True:
		print("match the centroids")
		print(dis_min)
		print(p[dis_min])
		print(dis)
	return dis_min

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

centroids = []
with open("centroids.txt", "r") as f:
	reader = csv.reader(f)
	cnt = 0
	row_ins = []
	for row in reader:
		if cnt % 5 != 0:
			row_ins.append(row)
		if cnt % 5 == 0 and cnt != 0:
			centroids.append(row_ins)
			row_ins = []
		cnt += 1
	centroids.append(row_ins)

file = open("complexity_cluster.txt").read().split("\n")
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
    result_cluster_wise = [[] for i in range(4)]
    file_name = str(measure_i) + ".txt"
    #file = open(file_name ,"w")
    count = 0
    temp = []
    for j in measures[measure_i]:
        if len(temp) == 4:
            n0 = count//4 - 1
            #print(n0)
            #sort the measures according to centroids
            if debug:
                print("----match the centroids----")
                print("n0", n0)
            temp_measure = [None for i in range(4)]
            #temp_seq = match(centroids[n0])
            temp_seq = 0
            for i_p in range(4):
                temp_measure[p[temp_seq][i_p]] = temp[i_p]
            for i_p in range(4):
                result_cluster_wise[i_p].append(temp_measure[i_p])
            temp = []
            if debug:
                print("----match the centroids----")

        if count == 400:
            #print(result_cluster_wise)
            with open("complexity_list.txt", "a") as f:
                writer = csv.writer(f)
                temp_list = [[] for i in range(4)]
                temp_list[0].append(measure_i)
                writer.writerows(temp_list)

            with open("complexity_cluster_format.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(result_cluster_wise)
                break

        temp.append(j)
        #print(temp)
        count +=1

