import csv
from sklearn.metrics import mutual_info_score
from sklearn import preprocessing
from sklearn.cluster import KMeans

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

#print(cluster_wise_distribution)

complexity_features = open("complexity_uci.txt").read().split("\n")
normal_features = open("normal_metafeatures_uci.txt").read().split("\n")
labels = open("dt_vs_rf.txt").read().split("\n")

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

for n0 in range(3):
	print("n0", n0)
	print("Cluster wise accuracy")
	print(cluster_wise_acc[n0])
	print("Cluster wise distribution")
	print(cluster_wise_distribution[n0])
	'''
	Y = []
	X = []
	row = [[] for i in range(2)]
	for cluster in range(4):
		for j in range(len(cluster_train[n0][cluster])):
			i = int(cluster_train[n0][cluster][j])
			X.append(cluster)
			Y.append(Y_temp[i])
	mu_in = mutual_info_score(X, Y)
	print("main")
	print(mu_in)
	row[0].append(mu_in)
	'''
	
	for cluster in range(4):
		print("cluster", cluster)
		X = []
		Y = []
		raw_data = []
		for j in range(len(cluster_train[n0][cluster])):
			i = int(cluster_train[n0][cluster][j])
			temp = []
			temp.append(X2_temp[i][4])
			temp.append(X2_temp[i][7])
			temp.append(X1_temp[i][5])
			temp.append(X1_temp[i][6])
			temp.append(X1_temp[i][9])
			temp.append(X1_temp[i][10])
			temp.append(X1_temp[i][12])
			temp.append(X1_temp[i][13])
			temp.append(X1_temp[i][19])
			temp.append(X1_temp[i][20])
			temp.append(X1_temp[i][21])
			X.append(temp)
			raw_data.append(temp)
			Y.append(Y_temp[i])

		k_mean = 3
		kmeans = KMeans(n_clusters = k_mean,n_init=1000, max_iter=10000).fit(raw_data) 
		labels = kmeans.labels_
		centroids = kmeans.cluster_centers_
		SSE = kmeans.inertia_
		flag = False

		
		count = [0,0,0]
		for i in labels:
			count[i] += 1
		for i in count:
			if i < 10:
				flag = True
		
		train_cluster = [[] for i in range(4)]
		for i in range(len(labels)):
			train_cluster[labels[i]].append(i)
		print(count)
		'''

		mu_in = mutual_info_score(labels, Y)
		#print(cluster)
		#print(mu_in)
		row[1].append(mu_in)
	print(row[1])

	with open("mutual_info.txt", "a") as f:
		writer = csv.writer(f)
		writer.writerows(row)
		'''
