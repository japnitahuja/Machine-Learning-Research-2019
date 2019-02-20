from sklearn import metrics
from s_dbw import S_Dbw
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv
from sklearn.metrics import mean_squared_error
from relative_criteria import Dunn_index, Davies_Bouldin, silhouette_index
import numpy

mse = mean_squared_error

def sse(true, pred):
    a = mse(true, pred, multioutput = 'raw_values')
    return sum(a)

complexity_features = open("complexity_uci.txt").read().split("\n")
normal_features = open("normal_metafeatures_uci.txt").read().split("\n")
labels = open("dt_vs_knn.txt").read().split("\n")

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

'''
#Read cluster Data -- test

test_data = open("centroids.txt","r")
cluster_test = []
count = 0

for i in test_data:
    i = i.strip()
    i = i.split(",")
    temp = []
    for x in i:
        temp.append(float(x))
    if count == 4:
        break
    if len(i) == 3:
        continue
    else:
        cluster_test.append(temp)
        count += 1

if Debug:
    for i in cluster_test:
        print(len(i))

#Reading the centroids

centroids = []

test_data = open("test_instance_label.txt","r")
cluster_test = []
count = 0

for i in test_data:
    i = i.strip()
    i = i.split(",")
    if count == 4:
        break
    if len(i) == 2:
        continue
    else:
        centroids.append(i)
        count += 1

if Debug:
    for i in centroids:
        print(len(i))
'''

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

#baseline measures
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

ch = metrics.calinski_harabaz_score(X, labels)
sil_score = metrics.silhouette_score(X, labels, metric='euclidean')

with open("cluster_analysis.txt", "a") as f:
            row = []
            row.append(ch)
            row.append(sil_score)
            writer = csv.writer(f)
            writer.writerow(row)

#print(len(cluster_train))
#print(len(cluster_train[0]))
for row_n in range(len(cluster_train)):
    ch_row = []
    sil_score_row = []
    dunn_row = []
    sil_in_row = []
    db_row = []
    for cluster in range(4):

        labels = []
        X = [] # features
        if Debug:
            print(row_n, cluster)

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

        #INTERNAL CLUSTER ANALYSIS
            
        #silhoutte score
        sil_score = metrics.silhouette_score(X, labels, metric='euclidean')

        if Debug:
            print("silhoutte score")
            print(sil_score)

        #calinksi harabasz
        ch = metrics.calinski_harabaz_score(X, labels)

        if Debug:
            print("calinksi harabasz")
            print(ch)

        dunn = Dunn_index(numpy.array(data_all))

        if Debug:
            print("dunn")
            print(dunn)

        sil_in = silhouette_index(numpy.array(data_all))

        if Debug:
            print("silhouette index")
            print(sil_in)

        db = Davies_Bouldin(numpy.array(data_all),numpy.array(centroids))

        if Debug:
            print("Davies_Bouldin")
            print(db)

        ch_row.append(ch)
        sil_score_row.append(sil_score)
        dunn_row.append(dunn)
        sil_in_row.append(sil_in)
        db_row.append(db)

    with open("cluster_analysis.txt", "a") as f:
        row = []
        row_n_list = []
        row_n_list.append(row_n)
        row.append(row_n_list)
        row.append(cluster_wise_acc[row_n])
        row.append(cluster_vs_baseline_result[row_n])
        row.append(sil_score_row)
        row.append(ch_row)
        row.append(dunn_row)
        row.append(sil_in_row)
        row.append(db_row)
        writer = csv.writer(f)
        writer.writerows(row)
'''
        sil_score_sum += sil_score
        ch_sum += ch

sil_score_sum /= 100
ch_sum /= 100

with open("cluster_analysis_mean.txt", "w") as f:
    writer = csv.writer(f)
    row = []
    row.append(sil_score_sum)
    row.append(ch_sum)
    writer.writerow(row)
    '''

