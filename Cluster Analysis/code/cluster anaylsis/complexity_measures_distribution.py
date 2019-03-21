import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv
import os

home = os.getcwd()
path = os.path.join(home, "complexity_distribution_graphs")


complexity_features = open("complexity_uci.txt").read().split("\n")
normal_features = open("normal_metafeatures_uci.txt").read().split("\n")
labels = open("dt_vs_rf.txt").read().split("\n")

file = open("cluster_wise_data.txt","w")
file.close()

Debug = True
sil_score_sum = 0
ch_sum = 0

n = 30000
x1 = [[0 for i in range(22)] for i in range(n)] #complexity
x2 = [[0 for i in range(19)] for i in range(n)] #normal
x3 = [[0 for i in range(14)] for i in range(n)] #tree
y = [None for i in range(n)]

#Complexity Features
flag = True
for i in complexity_features:
	if flag:
		flag = False
		continue

	if len(i) > 1:
		i = i.split(",")
		index = int(i[0])
		i = i[1:]

		if index >= 1155 and index <= 20629:
			continue

		for j in range(22):
			if i[j] != "inf":   
				x1[index][j] = float(i[j])

max_x1 = [-100000 for i in range(22)]
for i in range(len(x1)):
	for j in range(22):
		if x1[i][j] != "inf" and x1[i][j] >= max_x1[j]:
			max_x1[j] = float(x1[i][j])

for i in range(len(x1)):
	for j in range(22):
		if x1[i][j] == "inf":
			x1[i][j] = max_x1[j]*2

#Normal Features and Tree features
flag = False
for i in normal_features:
	if flag:
		flag = False
		continue
	
	if len(i) > 1:
		i = i.split(",")
		index = int(i[0])
		i = i[1:]


		if index >= 1155 and index <= 20629:
			continue

		for j in range(19):
			x2[index][j] = float(i[j])
			
		for j in range(19,33):
			x3[index][j-19] = float(i[j])

#scale the dataset
scaler = preprocessing.MinMaxScaler()
#preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
x1 = scaler.fit_transform(x1)
x2 = scaler.fit_transform(x2)
x3 = scaler.fit_transform(x3)

#labels
for i in labels:
	if len(i) > 1:
		i = i.split(",")
		index = int(i[0])

		if index >= 1155 and index <= 20629:
			continue
		i = int(i[1])
		y[index] = int(i)


X1_temp = []
X2_temp = []
X3_temp = []

Y_temp = []

for i in range(n):
	if y[i] != None:
		X1_temp.append(x1[i])
		X2_temp.append(x2[i])
		X3_temp.append(x3[i])
		Y_temp.append(y[i])



#Read cluster Data -- train
cluster_train = []
count = 0

with open("train_instance_label.txt", "r") as f:
	reader = csv.reader(f)
	cnt = 0
	temp = []
	cnt2 = 0
	for row in reader:
		if cnt % 2 == 1:
			temp.append(row)
			cnt2 += 1
		cnt += 1
		if cnt2 == 4:
			cluster_train.append(temp)
			temp = []
			cnt2 = 0

cluster_wise_acc = [[None for i in range(4)] for i in range(100)]
#read cluster_wise_accuracy
with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		cluster_wise_acc[int(row[0])*10+int(row[1])][int(row[2])] = float(row[3])

cluster_wise_distribution = []
with open("cluster_wise_label_distribution.txt", "r") as f:
	reader = csv.reader(f)
	cnt = 0
	cnt2 = 0
	temp = []
	for row in reader:
		if cnt % 2 == 1:
			temp.append(row)
			cnt2 += 1
			if cnt2 % 4 == 0:
				cluster_wise_distribution.append(temp)
				temp = []
				cnt2 = 0
		cnt += 1

for n0 in range(1):
	print("n0", n0)
	for feature in range(22):
		print("feature", feature)
		#baseline_fig_name = "baseline_" + str(n0) + '_' + str(feature) + ".png"
		#fig_name = str(n0) + '_' + str(feature) + ".png"
		baseline_x = []
		for cluster in range(4):
			X = []
			for j in range(len(cluster_train[n0][cluster])):
				i = int(cluster_train[n0][cluster][j])
				baseline_x.append(X1_temp[i][feature])
		'''
		with open("complexity_measures_average.txt", "a") as f:
			writer = csv.writer(f)
			row = []
			row.append(n0)
			writer.writerow(row)
			writer.writerow(cluster_wise_acc[n0])
			row = []
			row.append(sum(baseline_x)/len(baseline_x))
			writer.writerow(row)
		

		with open("complexity_measures_std_dev.txt", "a") as f:
			writer = csv.writer(f)
			row = []
			row.append(n0)
			writer.writerow(row)
			writer.writerow(cluster_wise_acc[n0])
			row = []
			row.append(np.std(baseline_x))
			writer.writerow(row)
		'''


		row_cluster = []
		row_cluster_std = []
		for cluster in range(4):
			fig_name = str(n0) + '_' + str(feature) + '_' + str(cluster) + ".png"
			X = []
			for j in range(len(cluster_train[n0][cluster])):
				i = int(cluster_train[n0][cluster][j])
				X.append(X1_temp[i][feature])
			''' 
			sns.distplot(X, color = "y")
			sns.distplot(baseline_x, color = "b")
			#plt.show()
			row = []
			row.append(cluster_wise_distribution[n0][cluster])
			row.append(cluster_wise_acc[n0][cluster])
			plt.title(row)
			os.chdir(path)
			plt.savefig(fig_name, dpi = 1000)
			plt.clf()
			os.chdir(home)
			'''
			row_cluster.append(sum(X)/len(X))
			row_cluster_std.append(np.std(X))
			'''
		with open("complexity_measures_average.txt", "a") as f:
			writer = csv.writer(f)
			writer.writerow(row_cluster)	
		
		with open("complexity_measures_std_dev.txt", "a") as f:
			writer = csv.writer(f)
			writer.writerow(row_cluster_std)
		'''
		with open("table.csv", "a") as f:
			writer = csv.writer(f)
			writer.writerow(row_cluster)
			writer.writerow(row_cluster_std)

