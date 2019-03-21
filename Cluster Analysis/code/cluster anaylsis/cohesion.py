import csv
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans

mse = mean_squared_error

def sse(true, pred):
	a = mse(true, pred, multioutput = 'raw_values')
	return sum(a)

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

cluster_test = []
with open("test_instance_label.txt", "r") as f:
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
			cluster_test.append(temp)
			temp = []
			cnt2 = 0

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

cluster_wise_acc = [[None for i in range(4)] for i in range(100)]
#read cluster_wise_accuracy
with open("cluster_wise_accuracy.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        cluster_wise_acc[int(row[0])*10+int(row[1])][int(row[2])] = float(row[3])

#read baseline per iteration
with open("hybrid_overall_acc.txt", "r") as f:
    reader = csv.reader(f)
    line = [row for row in reader]
baseline_acc = []
for row in line:
    if len(row) > 2:
        baseline_acc = row
        break

X_all_labels = []
#cluster_wise_sse = 0
for cluster in range(4):
	for j in range(len(cluster_train[0][cluster])):
		i = int(cluster_train[0][cluster][j])
		X_all_labels.append(i)
	for j in range(len(cluster_test[0][cluster])):
		i = int(cluster_test[0][cluster][j])
		X_all_labels.append(i) 
#divide the clusters into expertise label wise sub-clusters
label_wise_clusters = [[] for i in range(3)]
label_wise_sse = 0
for i in X_all_labels:
	label_wise_clusters[Y_temp[int(i)]].append(int(i))
for label_i in range(3):
	X = []
	Y = []
	label_wise_centroid = [0 for i in range(11)]
	for i in label_wise_clusters[label_i]:
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
		Y.append(Y_temp[i])
		for feature in range(11):
			label_wise_centroid[feature] += temp[feature]
	for feature in range(11):
		label_wise_centroid[feature] /= len(Y)
	X_centroid = [label_wise_centroid for i in range(len(Y))]
	label_wise_mse = mse(X, X_centroid)
	#print("label wise main", label_i)
	#print(label_wise_mse)
	label_wise_sse += sse(X, X_centroid)
print("main mse")
print(label_wise_sse/len(X_all_labels))


for n0 in range(3):
	print("n0", n0)
	print("Cluster wise accuracy")
	print(cluster_wise_acc[n0])

	'''
	Y = []
	X = []
	row = [[] for i in range(2)]
	for cluster in range(4):
		for j in range(len(cluster_train[n0][cluster])):
			i = int(cluster_train[n0][cluster][j])
			X.append(cluster)
			Y.append(Y_temp[i])
	main_mse = mse(X, Y)
	print("main cluster mse")
	print(main_mse)
	row[0].append(main_mse)
	'''

	'''
	#main label wise mse
	X_all_labels = []
	#cluster_wise_sse = 0
	for cluster in range(4):
		for j in range(len(cluster_train[n0][cluster])):
			i = int(cluster_train[n0][cluster][j])
			X_all_labels.append(i)
	#divide the clusters into expertise label wise sub-clusters
	label_wise_clusters = [[] for i in range(3)]
	label_wise_sse = 0
	for i in X_all_labels:
		label_wise_clusters[Y_temp[int(i)]].append(int(i))
	for label_i in range(3):
		X = []
		Y = []
		label_wise_centroid = [0 for i in range(11)]
		for i in label_wise_clusters[label_i]:
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
			Y.append(Y_temp[i])
			for feature in range(11):
				label_wise_centroid[feature] += temp[feature]
		for feature in range(11):
			label_wise_centroid[feature] /= len(Y)
		X_centroid = [label_wise_centroid for i in range(len(Y))]
		label_wise_mse = mse(X, X_centroid)
		#print("label wise main", label_i)
		#print(label_wise_mse)
		label_wise_sse += sse(X, X_centroid)
	print("main mse")
	print(label_wise_sse/len(X_all_labels))
	'''

	'''
		k_mean = 3
		kmeans = KMeans(n_clusters = k_mean,n_init=1000, max_iter=10000).fit(raw_data) 
		labels = kmeans.labels_
		centroids = kmeans.cluster_centers_
		SSE = kmeans.inertia_
		flag = False
		print("cluster wise mse", cluster)
		print(SSE/len(Y))
		row[1].append(SSE/len(Y))
		#cluster_wise_sse += SSE
		
	cluster_wise_mse = cluster_wise_sse / len(Y)
	print("cluster wise mse")
	print(cluster_wise_mse)
	row.append(cluster_wise_mse)
	'''

	#cluster wise label wise mse
	
	for cluster in range(4):
		print("cluster", cluster)
		#divide the clusters into expertise label wise sub-clusters
		label_wise_clusters = [[] for i in range(3)]
		label_wise_sse = 0
		for i in cluster_train[n0][cluster]:
			label_wise_clusters[Y_temp[int(i)]].append(int(i))

		for label_i in range(3):
			X = []
			Y = []
			label_wise_centroid = [0 for i in range(11)]
			for i in label_wise_clusters[label_i]:
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
				Y.append(Y_temp[i])
				for feature in range(11):
					label_wise_centroid[feature] += temp[feature]
			for feature in range(11):
				label_wise_centroid[feature] /= len(Y)
			X_centroid = [label_wise_centroid for i in range(len(Y))]
			#label_wise_mse = mse(X, X_centroid)
			#print("label wise per cluster", label_i)
			#print(label_wise_mse)
			label_wise_sse += sse(X, X_centroid)
		print("cluster wise mse")
		print(label_wise_sse/len(cluster_train[n0][cluster]))




	'''
	with open("cohesion_analysis.txt", "r") as f:
		writer = csv.writer(f)
		writer.writerow(row)
		'''
