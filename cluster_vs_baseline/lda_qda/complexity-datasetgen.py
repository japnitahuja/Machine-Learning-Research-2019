from sklearn import metrics
from s_dbw import S_Dbw
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv
import os
from sklearn.metrics import mean_squared_error
#from relative_criteria import Dunn_index, Davies_Bouldin, silhouette_index
import numpy

os.mkdir(os.path.join(os.getcwd(),"cluster_datasets"))
file = open("cluster_analysis.txt","w")
file.close()

mse = mean_squared_error

def sse(true, pred):
    a = mse(true, pred, multioutput = 'raw_values')
    return sum(a)

complexity_features = open("complexity_measures.txt").read().split("\n")
normal_features = open("Classical-Decision_Tree.txt").read().split("\n")
labels = open("lda_vs_qda.txt").read().split("\n")

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

        if index >= 1155 and index <= 20629:
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

X=[]
labels = []

for i in range(len(Y_temp)):
    i = int(i)
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
    labels.append(Y_temp[i])


#print(len(cluster_train))
#print(len(cluster_train[0]))

#n0 = len(cluster_train)
n0 = 100
count = 0
for row_n in range(n0):
    ch_row = []
    sil_score_row = []
    dunn_row = []
    sil_in_row = []
    db_row = []
    for cluster in range(4):

        if Debug:
            print("cluster")
            print(cluster)
            print("accuracy")
            print(cluster_wise_acc[row_n][cluster])
            print(cluster_vs_baseline_result[row_n][cluster])
            
        labels = []
        X = [] # features
        if Debug:
            print(row_n, cluster)

        exp_space_distribution_train = [0,0,0]
        exp_space_distribution_test = [0,0,0]

        data_all = [] #features + label

        for i in cluster_train[row_n][cluster]:
            i = int(i)
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
            X.append(temp[:])
            temp.append(Y_temp[i])
            data_all.append(temp[:])
            labels.append(Y_temp[i])
            exp_space_distribution_train[Y_temp[i]] += 1

        file = open(os.path.join("cluster_datasets",str(count) + ".txt"), "w")

        print(len(data_all))

        for i in data_all:
            temp = ""
            for j in i:
                temp += str(j)
                temp += ","
            temp = temp[:-1]
            temp += "\n"
            file.write(temp)
        count += 1
        file.close()
print("count inf", cnt_inf)


