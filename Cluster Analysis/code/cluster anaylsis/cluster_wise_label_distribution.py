import csv
from sklearn import preprocessing

complexity_features = open("complexity_uci.txt").read().split("\n")
normal_features = open("normal_metafeatures_uci.txt").read().split("\n")
labels = open("dt_vs_knn.txt").read().split("\n")

file = open("cluster_wise_data.txt","w")
file.close()

Debug = True

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

for row_n in range(len(cluster_train)):
	for cluster in range(4):
		labels = []
		if Debug:
			print(row_n, cluster)

		for i in cluster_train[row_n][cluster]:
			labels.append(Y_temp[int(i)])

		distribution = [0 for i in range(3)]
		for i in labels:
			distribution[int(i)] += 1

		with open("cluster_wise_label_distribution.txt", "a") as f:
			writer = csv.writer(f)
			row = []
			row.append(row_n)
			row.append(cluster)
			writer.writerow(row)
			writer.writerow(distribution)

