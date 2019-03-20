from sklearn import metrics
from s_dbw import S_Dbw
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv
import os
from sklearn.metrics import mean_squared_error
#from relative_criteria import Dunn_index, Davies_Bouldin, silhouette_index
import numpy

complexity2_path = os.path.join(os.getcwd(),"complexity2")
if not os.path.isdir(complexity2_path):
    os.mkdir(complexity2_path)
file = open("cluster_analysis.txt","w")
file.close()

home = os.getcwd()
complexity2 = os.path.join(home, "complexity2")
os.chdir(complexity2)
cluster_path = os.path.join(complexity2, "cluster_dataset")
if not os.path.isdir(cluster_path):
    os.mkdir(cluster_path)
baseline_path = os.path.join(complexity2, "baseline_dataset")
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
os.chdir(home)

mse = mean_squared_error

def sse(true, pred):
    a = mse(true, pred, multioutput = 'raw_values')
    return sum(a)

complexity_features = open("complexity_measures.txt").read().split("\n")
normal_features = open("Classical-Decision_Tree.txt").read().split("\n")
labels = open("dt_vs_lda.txt").read().split("\n")

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
cnt_inf = 0
for i in complexity_features:
    if flag:
        flag = False
        continue

    if len(i) > 1:
        i = i.split(",")
        index = int(i[0])
        i = i[1:]

        if index >= 400 and index < 600:
            continue

        for j in range(22):
            if i[j] == "inf":
                cnt_inf += 1
            if i[j] != "inf":   
                x1[index][j] = float(i[j])
'''
max_x1 = [-100000 for i in range(22)]
for i in range(len(x1)):
    for j in range(22):
        if x1[i][j] != "inf" and x1[i][j] >= max_x1[j]:
            max_x1[j] = float(x1[i][j])

for i in range(len(x1)):
    for j in range(22):
        if x1[i][j] == "inf":
            x1[i][j] = max_x1[j]*2
'''

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


        if index >= 400 and index < 600:
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

        if index >= 400 and index < 600:
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


#Read cluster Data -- test
cluster_test = []
count = 0

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

#Read the best cluster meta feature for cluster and baseline
with open("best_cluster_metafeature_cluster.txt", "r") as f:
    reader = csv.reader(f)
    best_cluster_metafeature_cluster = [row for row in reader]

with open("best_cluster_metafeature_baseline.txt", "r") as f:
    reader = csv.reader(f)
    best_cluster_metafeature_baseline = [row for row in reader]


#Reading the centroids
centroids = []
with open("centroids.txt", "r") as f:
    reader = csv.reader(f)
    cnt = 0
    for row in reader:
        if cnt % 2 == 1:
            centroids.append(row)
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
#print(baseline_acc)

cluster_vs_baseline_result = [[None for i in range(4)] for j in range(100)]
#compare cluster wise accuracy with baseline accuracy
for i in range(100):
    for j in range(4):
        if float(baseline_acc[i]) < cluster_wise_acc[i][j]:
            cluster_vs_baseline_result[i][j] = 1
        else: 
            cluster_vs_baseline_result[i][j] = 0
#print(cluster_vs_baseline_result)

best_metafeature = {"0":X1_temp, "1":X2_temp, "2":X3_temp}

#print(len(cluster_train))
#print(len(cluster_train[0]))

print(len(best_cluster_metafeature_cluster))
print(len(best_cluster_metafeature_baseline))
print(len(best_metafeature["0"]))

#n0 = len(cluster_train)
n0 = 100
count = 0
for row_n in range(n0):
    #cluster wise
    for cluster in range(4):

        if Debug:
            print("cluster", cluster)
            #print("accuracy")
            #print(cluster_wise_acc[row_n][cluster])
            #print(cluster_vs_baseline_result[row_n][cluster])
            
        if Debug:
            print(row_n, cluster)

        #print(best_cluster_metafeature_cluster)
        #print(row_n, cluster)
        best_metafeature_label = best_cluster_metafeature_cluster[row_n][cluster]

        for i in cluster_train[row_n][cluster]:
            i = int(i)
            if i >= 600:
                i -= 200
            temp = []
            for j in range(len(best_metafeature[best_metafeature_label][0])):
                #print(best_metafeature_label,i,j)
                temp.append(best_metafeature[best_metafeature_label][i][j])
            temp.append(Y_temp[i])

            os.chdir(cluster_path)
            new_file = str(count) + ".txt"
            with open(new_file, "a") as f:
                writer =csv.writer(f)
                writer.writerow(temp)
        count += 1

    #baseline
    best_metafeature_label = best_cluster_metafeature_baseline[row_n][0]

    #print(best_cluster_metafeature_baseline)
    for cluster in range(4):
        for i in cluster_train[row_n][cluster]:
            i = int(i)
            if i >= 600:
                i -= 200
            temp = []
            for j in range(len(best_metafeature[best_metafeature_label])):
                temp.append(best_metafeature[best_metafeature_label][i][cluster])
            temp.append(Y_temp[i])

            os.chdir(baseline_path)
            new_file = str(row_n) + ".txt"
            with open(new_file, "a") as f:
                writer =csv.writer(f)
                writer.writerow(temp)

#print("count inf", cnt_inf)
            

 


