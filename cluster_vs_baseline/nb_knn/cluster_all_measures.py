import csv
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mutual_info_score

flag_separation2 = 0

debug = 0

mse = mean_squared_error

def sse(true, pred):
	a = mse(true, pred, multioutput = 'raw_values')
	return sum(a)

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


complexity_features = open("complexity_measures.txt").read().split("\n")
normal_features = open("Classical-Decision_Tree.txt").read().split("\n")
labels = open("nb_vs_knn.txt").read().split("\n")

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
	centroids.append(row_ins)

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



result_baseline = [[] for i in range(100)]
result_cluster_wise = [[[] for j in range(4)] for i in range(100)]
'''
mutual infomation
silhouette score
cohesion (compleixty cluster)
separation 1(^)
separation 2(^)
cohesion (expertise cluster)
separation 1 (^)
separation 2 (^)
'''

#print(centroids)
for n0 in range(100):
	print("n0", n0)
	#print("Cluster wise accuracy")
	#print(cluster_wise_acc[n0])

	'''
	----- baseline complexity based cluster analysis -----
	'''
	print("baseline complexity cluster analysis")
	X = []
	Y = []
	raw_data = []
	cluster_labels = []
	for cluster in range(4):
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
			cluster_labels.append(cluster)
			raw_data.append(temp)
			Y.append(Y_temp[i])
	k_mean = 3
	kmeans = KMeans(n_clusters = k_mean,n_init=1000, max_iter=10000).fit(raw_data) 
	labels = kmeans.labels_
	centroids_kmeans = kmeans.cluster_centers_
	SSE = kmeans.inertia_
	flag = False

	#mutual infomation
	mu_in = mutual_info_score(cluster_labels, Y)
	result_baseline[n0].append(mu_in)
	if debug:
		print("mu_in")
		print(mu_in)

	#silhoutte score
	sil_score = metrics.silhouette_score(X, Y, metric='euclidean')
	result_baseline[n0].append(sil_score)
	if debug:
		print("sil_score")
		print(sil_score)


	#cohesion
	cluster_labels = []
	label_wise_sse = 0
	cnt_ins = 0
	for cluster in range(4):
		X = []
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
			cluster_labels.append(cluster)
			cnt_ins += 1
		X_centroid = [centroids[n0][cluster] for k in range(len(cluster_train[n0][cluster]))]
		X = np.array(X).astype(np.float64)
		X_centroid = np.array(X_centroid).astype(np.float64)
		label_wise_sse += sse(X, X_centroid)

	result_baseline[n0].append(label_wise_sse/cnt_ins)
	if debug:
		print("cohesion")
		print(label_wise_sse/cnt_ins)


	#separation
	min_separation = None
	dis = 0
	for cluster in range(4):
		#separation(max dis between centroids)
		for cluster_2 in range(cluster+1, 3):
			dis_temp = 0 
			for feature in range(11):
				dis_temp += abs(float(centroids[n0][cluster][feature]) - float(centroids[n0][cluster_2][feature]))
				#print("abs", abs(centroids_kmeans[re_cluster][feature] - centroids_kmeans[re_cluster_2][feature]))
				#print("dis_temp", dis_temp)
			if min_separation == None: 
				min_separation = dis_temp
			elif dis_temp > min_separation:
				min_separation = dis_temp

		#separation(dis between instances of one cluster to the rest)
		if flag_separation2:
			for t_cluster in range(4):
				#print("t_cluster", t_cluster)
				if t_cluster != cluster:
					X1 = []
					Y1 = []
					raw_data1 = []
					for j in range(len(cluster_train[n0][t_cluster])):
						i = int(cluster_train[n0][t_cluster][j])
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
						X1.append(temp)
						raw_data1.append(temp)
						Y1.append(Y_temp[i])
					for i in range(len(X)):
						for j in range(len(X1)):
							for feature in range(11): 
								dis += abs(float(X[i][feature]) - float(X1[j][feature]))

	#print(min_separation) 
	result_baseline[n0].append(min_separation)

	#print("separation")
	#print(dis)
	result_baseline[n0].append(dis)

	if debug:
		print("separation 1")
		print(min_separation)
		print("separation 2")
		print(dis)

	'''
	----- baseline complexity based cluster analysis end -----
	'''


	''' 
	----- baseline expertise cluster analysis -----
	'''
	print("baseline expertise cluster analysis")
	#divide the clusters into expertise label wise sub-clusters
	label_wise_clusters = [[] for i in range(3)]
	label_wise_sse = 0
	for cluster in range(4):
		for i in cluster_train[n0][cluster]:
			label_wise_clusters[Y_temp[int(i)]].append(int(i))

	#cohesion
	for label_i in range(3):
		X = []
		Y = []
		label_wise_centroid = [[0 for i in range(11)] for j in range(3)]
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
				label_wise_centroid[label_i][feature] += temp[feature]
		for feature in range(11):
			label_wise_centroid[label_i][feature] /= len(Y)
		X_centroid = [label_wise_centroid[label_i] for i in range(len(Y))]
		label_wise_sse += sse(X, X_centroid)

	result_baseline[n0].append(label_wise_sse/len(cluster_train[n0][cluster]))
	if debug:
		print("cohesion")
		print(label_wise_sse/len(cluster_train[n0][cluster]))


	#separation 
	min_separation = None
	dis = 0
	for re_cluster in range(3):
		#separation(max distance between centroids) 
		for re_cluster_2 in range(3):
			if re_cluster_2 == re_cluster:
				continue
			dis_temp = 0
			for feature in range(11):
				dis_temp += abs(label_wise_centroid[re_cluster][feature] - label_wise_centroid[re_cluster_2][feature])
				#print("abs", abs(centroids_kmeans[re_cluster][feature] - centroids_kmeans[re_cluster_2][feature]))
				#print("dis_temp", dis_temp)
			if min_separation == None: 
				min_separation = dis_temp
			elif dis_temp > min_separation:
				min_separation = dis_temp
	

		#separation(sum distance between instances from one clusters to the rest)
		if flag_separation2:
			for t_cluster in range(4):
				#print("t_cluster", t_cluster)
				if t_cluster != re_cluster:
					X1 = []
					Y1 = []
					raw_data1 = []
					for j in range(len(cluster_train[n0][t_cluster])):
						i = int(cluster_train[n0][t_cluster][j])
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
						X1.append(temp)
						raw_data1.append(temp)
						Y1.append(Y_temp[i])
					for i in range(len(X)):
						for j in range(len(X1)):
							for feature in range(11): 
								dis += (float(X[i][feature]) - float(X1[j][feature]))**2

	#print(min_separation) 
	result_baseline[n0].append(min_separation)

	#print("separation")
	#print(dis)
	result_baseline[n0].append(dis)
	if debug:
		print("separation 1")
		print(min_separation)
		print("separation 2")
		print(dis)


	'''
	----- baseline expertise cluster analysis end -----
	'''

	print("cluster wise analysis")
	for cluster in range(4):
		print("cluster", cluster)

		'''
		----- meta features based sub-clusters (kMeans) -----
		'''
		print("complexity cluster sub cluster analysis")
		sub_cluster = [[] for i in range(3)]

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
		centroids_kmeans = kmeans.cluster_centers_
		SSE = kmeans.inertia_
		flag = False

		#mutual infomation
		mu_in = mutual_info_score(labels, Y)
		result_cluster_wise[n0][cluster].append(mu_in)
		if debug:
			print("mu_in")
			print(mu_in)

		#silhoutte score
		sil_score = metrics.silhouette_score(X, labels, metric='euclidean')
		result_cluster_wise[n0][cluster].append(sil_score)
		if debug:
			print("sil_score")
			print(sil_score)


		#create a list store all instance index for each sub cluster
		for j in range(len(cluster_train[n0][cluster])):
			#print(cluster_train[n0][cluster][j])
			i = int(cluster_train[n0][cluster][j])
			sub_cluster[labels[j]].append(i)

		#cohesion
		cluster_labels = []
		label_wise_sse = 0
		cnt_ins = 0
		for re_cluster in range(3):
			X = []
			for j in range(len(sub_cluster[re_cluster])):
				i = int(sub_cluster[re_cluster][j])
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
				cnt_ins += 1
			X_centroid = [centroids_kmeans[re_cluster] for k in range(len(sub_cluster[re_cluster]))]
			label_wise_sse += sse(X, X_centroid)

		result_cluster_wise[n0][cluster].append(label_wise_sse/cnt_ins)
		if debug:
			print("cohesion")
			print(label_wise_sse/cnt_ins)


		#separation
		min_separation = None
		dis = 0
		for re_cluster in range(3): 
			#separation(max distance between centroids)
			for re_cluster_2 in range(re_cluster+1, 3):
				dis_temp = 0
				for feature in range(11):
					dis_temp += abs(centroids_kmeans[re_cluster][feature] - centroids_kmeans[re_cluster_2][feature])
					#print("abs", abs(centroids_kmeans[re_cluster][feature] - centroids_kmeans[re_cluster_2][feature]))
					#print("dis_temp", dis_temp)
				if min_separation == None: 
					min_separation = dis_temp
				elif dis_temp > min_separation:
					min_separation = dis_temp

			#separation(sum distance between instances from one clusters to the rest)
			if flag_separation2:
				for t_cluster in range(3):
					#print("t_cluster", t_cluster)
					if t_cluster != re_cluster:
						X1 = []
						Y1 = []
						raw_data1 = []
						for j in range(len(sub_cluster[t_cluster])):
							i = int(sub_cluster[t_cluster][j])
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
							X1.append(temp)
							raw_data1.append(temp)
							Y1.append(Y_temp[i])
						for i in range(len(X)):
							for j in range(len(X1)):
								for feature in range(11): 
									dis += (float(X[i][feature]) - float(X1[j][feature]))**2
		
		#print(min_separation) 
		result_cluster_wise[n0][cluster].append(min_separation)

		#print("separation")
		#print(dis)
		result_cluster_wise[n0][cluster].append(dis)
		if debug:
			print("separation 1")
			print(min_separation)
			print("separation 2")
			print(dis)


		'''
		----- meta features based sub-clusters (kMeans) end -----
		'''


		'''
		----- expertise sub-clusters ----
		'''
		print("expertise sub cluster analysis")
		#divide the clusters into expertise label wise sub-clusters
		label_wise_clusters = [[] for i in range(3)]
		label_wise_sse = 0
		for i in cluster_train[n0][cluster]:
			label_wise_clusters[Y_temp[int(i)]].append(int(i))
		#print(len(label_wise_clusters[0]))
		#print(len(label_wise_clusters[1]))
		#print(len(label_wise_clusters[2]))

		#cohesion
		for label_i in range(3):
			X = []
			Y = []
			label_wise_centroid = [[0 for i in range(11)] for j in range(3)]
			if len(label_wise_clusters[label_i]) == 0:
				continue
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
					label_wise_centroid[label_i][feature] += temp[feature]
			for feature in range(11): 
				#print(len(Y))
				label_wise_centroid[label_i][feature] /= len(Y)
			X_centroid = [label_wise_centroid[label_i] for i in range(len(Y))]
			label_wise_sse += sse(X, X_centroid)

		result_cluster_wise[n0][cluster].append(label_wise_sse/len(cluster_train[n0][cluster]))
		if debug:
			print("cohesion")
			print(label_wise_sse/len(cluster_train[n0][cluster]))


		#separation 
		min_separation = None
		dis = 0
		for re_cluster in range(3):
			#separation(max distance between centroids) 
			for re_cluster_2 in range(3):
				if re_cluster_2 == re_cluster:
					continue
				dis_temp = 0
				for feature in range(11):
					dis_temp += abs(label_wise_centroid[re_cluster][feature] - label_wise_centroid[re_cluster_2][feature])
					#print("abs", abs(centroids_kmeans[re_cluster][feature] - centroids_kmeans[re_cluster_2][feature]))
					#print("dis_temp", dis_temp)
				if min_separation == None: 
					min_separation = dis_temp
				elif dis_temp > min_separation:
					min_separation = dis_temp
		

			#separation(sum distance between instances from one clusters to the rest)
			if flag_separation2:
				for t_cluster in range(4):
					#print("t_cluster", t_cluster)
					if t_cluster != re_cluster:
						X1 = []
						Y1 = []
						raw_data1 = []
						for j in range(len(cluster_train[n0][t_cluster])):
							i = int(cluster_train[n0][t_cluster][j])
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
							X1.append(temp)
							raw_data1.append(temp)
							Y1.append(Y_temp[i])
						for i in range(len(X)):
							for j in range(len(X1)):
								for feature in range(11): 
									dis += (float(X[i][feature]) - float(X1[j][feature]))**2

		#print(min_separation) 
		result_cluster_wise[n0][cluster].append(min_separation)

		#print("separation")
		#print(dis)
		result_cluster_wise[n0][cluster].append(dis)

		if debug:
			print("separation 1")
			print(min_separation)
			print("separation 2")
			print(dis)

		'''
		----- expertise sub-clusters end ----
		'''
	if debug:
		print("----match the centroids----")
		print("n0", n0)
	temp = result_cluster_wise[n0][:]
	temp_measure = [None for i in range(4)]
	temp_seq = match(centroids[n0])
	for i_p in range(4):
		temp_measure[p[temp_seq][i_p]] = temp[i_p]
	result_cluster_wise[n0] = temp_measure[:]
	if debug:
		print("----match the centroids----")



with open("result_baseline.txt", "w") as f:
	writer = csv.writer(f)
	'''
	for i in range(8):
		temp = []
		for j in range(100):
			temp.append(result_baseline[j][i])
		writer.writerow(temp)
		'''
	writer.writerows(result_baseline)


with open("result_cluster_wise.txt", "w") as f:
	writer = csv.writer(f)
	'''
	for i in range(8):
		for cluster in range(4):
			temp = []
			for j in range(100):
				temp.append(result_cluster_wise[j][i][cluster])
			writer.writerow(temp)
			'''
	for i in range(100):
		for j in range(4):
			writer.writerow(result_cluster_wise[i][j])









