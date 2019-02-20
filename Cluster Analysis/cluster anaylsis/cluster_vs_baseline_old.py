import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import os
from scipy import stats

complexity_features = open("complexity_uci.txt").read().split("\n")
normal_features = open("normal_metafeatures_uci.txt").read().split("\n")
labels = open("dt_vs_rf.txt").read().split("\n")

file = open("cluster_wise_data.txt","w")
file.close()


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
            if i[j] == "inf":
                i[j] = 100000
                      
            x1[index][j] = float(i[j])

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

accuracy_baseline = []
accuracy_cluster = []

#outer 10x10cv
for n0 in range(10):
    print("outer")
    print(n0)

    label_wise = [[] for i in range(3)]

    for i in range(len(Y_temp)):
        index = Y_temp[i]
        label_wise[index].append(i)
        
    label_wise_all = []
    for i in label_wise:
        np.random.shuffle(i)
        for j in i:
            label_wise_all.append(j)

    folds = [[] for i in range(10)]

    for j in range(len(label_wise_all)):
        temp = j % 10
        temp = folds[temp]
        temp.append(label_wise_all[j])

    for test_fold in range(10):
        print(test_fold)
        #overarching training and test sets 
        train = []
        test = []

        for fold in range(10):
            temp = folds[fold]
            if fold == test_fold:
                for i in temp:
                    test.append(i)
            else:
                for i in temp:
                    train.append(i)

        raw_data = []

        print("TRAIN AND TEST")
        print(len(train),len(test))

        for i in train:
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
            raw_data.append(temp)

        #CLUSTER BASED MODEL
        #clustering the training set
        flag = True
        
        while flag:
            k_mean = 4
            kmeans = KMeans(n_clusters = k_mean,n_init=1000, max_iter=10000).fit(raw_data) 
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            flag = False
            
            count = [0,0,0,0]
            for i in labels:
                count[i] += 1
            for i in count:
                if i < 10:
                    flag = True
            
        
        best_cluster_metafeature = [0,0,0,0]
        best_cluster_k = [0,0,0,0]
        best_cluster_accuracy = [[] for i in range(4)]

        print("clusterbased")
        #internal 10cv
        for cluster in range(k_mean):
            cluster_folds = [[] for i in range(10)]
            
            X1, X2 , X3, Y = [], [], [], []

            count = 0
            for i in train:
                if labels[count] == cluster:
                    X1.append(X1_temp[i])
                    X2.append(X2_temp[i])
                    X3.append(X3_temp[i])
                    Y.append(Y_temp[i])
                count += 1
                    
            
            print("cluster: " + str(cluster))
            print(len(Y))

            #choosing the best model
            accuracy1 = [[] for i in range(13)]
            accuracy2 = [[] for i in range(13)]
            accuracy3 = [[] for i in range(13)]

            
            cluster_label_wise = [[] for i in range(3)]
        
            for i in range(len(Y)):
                index = Y[i]
                cluster_label_wise[index].append(i)

            cluster_label_wise_all = []
            for i in cluster_label_wise:
                np.random.shuffle(i)
                for j in i:
                    cluster_label_wise_all.append(j)

            #division into 10 folds
            for j in range(len(cluster_label_wise_all)):
                temp = j % 10
                temp = cluster_folds[temp]
                temp.append(cluster_label_wise_all[j])

            
            for cluster_test_fold in range(10):
                #complexity
                train_x_1 = []
                test_x_1 = []

                #normal
                train_x_2 = []
                test_x_2 = []


                #tree
                train_x_3 = []
                test_x_3 = []


                train_y = []
                test_y = []
         

                for fold in range(10):
                    temp = cluster_folds[fold]
                    if fold == cluster_test_fold:
                        for i in temp:
                            test_x_1.append(X1[i])
                            test_x_2.append(X2[i])
                            test_x_3.append(X3[i])
                            test_y.append(Y[i])

                    else:
                        for i in temp:
                            train_x_1.append(X1[i])
                            train_x_2.append(X2[i])
                            train_x_3.append(X3[i])
                            train_y.append(Y[i])

                            
                for k in range(2,12):
                    if k > len(train_x_1):
                        continue
                    classifier1 = KNeighborsClassifier(n_neighbors=k)
                    classifier2 = KNeighborsClassifier(n_neighbors=k)
                    classifier3 = KNeighborsClassifier(n_neighbors=k)
                    
                    classifier1.fit(train_x_1,train_y)
                    classifier2.fit(train_x_2,train_y)
                    classifier3.fit(train_x_3,train_y)
                    
                    accuracy1[k].append(classifier1.score(test_x_1,test_y))
                    accuracy2[k].append(classifier2.score(test_x_2,test_y))
                    accuracy3[k].append(classifier3.score(test_x_3,test_y))

            accuracy_mean_1 = [0 for i in range(13)]
            accuracy_mean_2 = [0 for i in range(13)]
            accuracy_mean_3 = [0 for i in range(13)]

            for i in range(13):
                accuracy_mean_1[i] = sum(accuracy1[i])
                accuracy_mean_2[i] = sum(accuracy2[i])
                accuracy_mean_3[i] = sum(accuracy3[i])

            #choosing the best model for this cluster after 10cv
            best_accuracy = [0,0,0]
            best_k = [0,0,0]
            
            best_accuracy[0] = max(accuracy_mean_1)
            best_k[0] = accuracy_mean_1.index(max(accuracy_mean_1))

            best_accuracy[1] = max(accuracy_mean_2)
            best_k[1] = accuracy_mean_2.index(max(accuracy_mean_2))
            
            best_accuracy[2] = max(accuracy_mean_3)
            best_k[2] = accuracy_mean_3.index(max(accuracy_mean_3))
            
            #saving the best model for the cluster info
            best_cluster_metafeature[cluster] = best_accuracy.index(max(best_accuracy))
            best_cluster_k[cluster] = best_k[best_cluster_metafeature[cluster]]
              
            if best_cluster_metafeature[cluster] == 0:
                best_cluster_accuracy[cluster] = accuracy1[best_cluster_k[cluster]]
                
                
            elif best_cluster_metafeature[cluster] == 1:
                best_cluster_accuracy[cluster] = accuracy2[best_cluster_k[cluster]]
                
                
            elif best_cluster_metafeature[cluster] == 2:
                best_cluster_accuracy[cluster] = accuracy3[best_cluster_k[cluster]]
                                         

            #to store the best model accuracy and proportion of training set
            file = open("best_model_accuracy_" + str(cluster) + ".txt","a")
            file.write(str(max(best_accuracy)) + "," + str(best_cluster_k[cluster]) + "\n")
                                                             
            proportion = [0,0,0]
            for i in Y:
                proportion[i] += 1
                                                             
            file.write(str(proportion[0]) + "," + str(proportion[1]) + "," + str(proportion[2])+ "\n")

            file.close()
   
        print("baseline")
        #BASELINE MODEL
        best_baseline_metafeature = 0
        best_baseline_k = 0
        best_baseline_accuracy = []

        accuracy1 = [[] for i in range(13)]
        accuracy2 = [[] for i in range(13)]
        accuracy3 = [[] for i in range(13)]

        X1, X2 , X3, Y = [], [], [], []
                
        for i in range(len(Y_temp)):
            if i in train:
                X1.append(X1_temp[i])
                X2.append(X2_temp[i])
                X3.append(X3_temp[i])
                Y.append(Y_temp[i])

        baseline_label_wise = [[] for i in range(3)]

        for i in range(len(Y)):
            index = Y[i]
            baseline_label_wise[index].append(i)
            
        baseline_label_wise_all = []
        for i in baseline_label_wise:
            np.random.shuffle(i)
            for j in i:
                baseline_label_wise_all.append(j)

        baseline_folds = [[] for i in range(10)]

        for j in range(len(baseline_label_wise_all)):
            temp = j % 10
            temp = baseline_folds[temp]
            temp.append(baseline_label_wise_all[j])
           

        for baseline_test_fold in range(10):
            #complexity
            train_x_1 = []
            test_x_1 = []

            #normal
            train_x_2 = []
            test_x_2 = []


            #tree
            train_x_3 = []
            test_x_3 = []


            train_y = []
            test_y = []
     

            for fold in range(10):
                temp = baseline_folds[fold]
                if fold == baseline_test_fold:
                    for i in temp:
                        test_x_1.append(X1[i])
                        test_x_2.append(X2[i])
                        test_x_3.append(X3[i])
                        test_y.append(Y[i])

                else:
                    for i in temp:
                        train_x_1.append(X1[i])
                        train_x_2.append(X2[i])
                        train_x_3.append(X3[i])
                        train_y.append(Y[i])

            for k in range(2,12):
                if k > len(train_x_1):
                    continue
                
                classifier1 = KNeighborsClassifier(n_neighbors=k)
                classifier2 = KNeighborsClassifier(n_neighbors=k)
                classifier3 = KNeighborsClassifier(n_neighbors=k)
                
                classifier1.fit(train_x_1,train_y)
                classifier2.fit(train_x_2,train_y)
                classifier3.fit(train_x_3,train_y)
                
                acc1 = classifier1.score(test_x_1,test_y)
                acc2 = classifier2.score(test_x_2,test_y)
                acc3 = classifier3.score(test_x_3,test_y)

                accuracy1[k].append(acc1)
                accuracy2[k].append(acc2)
                accuracy3[k].append(acc3)

        accuracy_mean_1 = [0 for i in range(13)]
        accuracy_mean_2 = [0 for i in range(13)]
        accuracy_mean_3 = [0 for i in range(13)]
        
        for i in range(13):
            accuracy_mean_1[i] = sum(accuracy1[i])
            accuracy_mean_2[i] = sum(accuracy2[i])
            accuracy_mean_3[i] = sum(accuracy3[i])

        best_accuracy = [0,0,0]
        best_k = [0,0,0]
        
        best_accuracy[0] = max(accuracy_mean_1)
        best_k[0] = accuracy_mean_1.index(max(accuracy_mean_1))

        best_accuracy[1] = max(accuracy_mean_2)
        best_k[1] = accuracy_mean_2.index(max(accuracy_mean_2))
        
        best_accuracy[2] = max(accuracy_mean_3)
        best_k[2] = accuracy_mean_3.index(max(accuracy_mean_3))

        #saving the best model for the cluster info
        best_baseline_metafeature= best_accuracy.index(max(best_accuracy))
        best_baseline_k= best_k[best_baseline_metafeature]
                                     
        if best_baseline_metafeature == 0:
            best_baseline_accuracy = accuracy1[best_baseline_k]
            
        elif best_baseline_metafeature == 1:
            best_baseline_accuracy = accuracy2[best_baseline_k]
            
        elif best_baseline_metafeature == 2:
            best_baseline_accuracy = accuracy3[best_baseline_k]

        #FINAL EVALUATION
        #Test
        raw_data_test = []
        test_y = []

        for i in test:
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
            raw_data_test.append(temp)
            test_y.append(Y_temp[i])

        
        test_cluster = []
        
        #alloting cluster labels to test data
        for i in raw_data_test:
            dist = []
            for j in centroids:
                temp_dist = 0
                for k in range(11):
                    temp_dist += abs(pow(i[k],2)-pow(j[k],2))
                dist.append(temp_dist)
            test_cluster.append(dist.index(min(dist)))

        cluster_wise_test = [[] for i in range(4)]
        cluster_wise_true_values = [[] for i in range(4)]
        
        for i in range(len(test)):
            cluster_wise_test[test_cluster[i]].append(test[i])
            cluster_wise_true_values[test_cluster[i]].append(test_y[i])

        #Train
        raw_data_train = []
        train_cluster = []

        for i in train:
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
            raw_data_train.append(temp)

        cluster_wise_train = [[] for i in range(4)]

        #allotting cluster labels to train data
        for i in raw_data_train:
            dist = []
            for j in centroids:
                temp_dist = 0
                for k in range(11):
                    temp_dist += abs(pow(i[k],2)-pow(j[k],2))
                dist.append(temp_dist)
            train_cluster.append(dist.index(min(dist)))

        for i in range(len(train)):
            cluster_wise_train[train_cluster[i]].append(train[i])
            
        for i in cluster_wise_true_values:
            count = [0,0,0]
            for j in i:
                count[j] += 1
            file = open("cluster_wise_data.txt","a")
            temp = ""
            for i in count:
                temp += str(i)+","
            file.write(temp[:-1] + "\n")
        file.close()

        metafeatures = [X1_temp,X2_temp,X3_temp]
        correct_cluster = [0,0,0,0]
        correct_baseline = [0,0,0,0]
        total = [0,0,0,0]

        
        for cluster in range(4):
            #Cluster
            
            cluster_test = cluster_wise_test[cluster]
            y_test = cluster_wise_true_values[cluster]

            cluster_train = cluster_wise_train[cluster]

            mf_set_cluster = metafeatures[best_cluster_metafeature[cluster]]
            k_cluster = best_cluster_k[cluster]

            x_train_cluster = []
            y_train_cluster = []
            
            for i in cluster_train:
                x_train_cluster.append(mf_set_cluster[i])
                y_train_cluster.append(Y_temp[i])

            x_test_cluster = []

            for i in cluster_test:
                x_test_cluster.append(mf_set_cluster[i])

            cluster_classifier = KNeighborsClassifier(n_neighbors=k_cluster)
            cluster_classifier.fit(x_train_cluster,y_train_cluster)

            #Baseline

            x_train_baseline = []
            y_train_baseline = []

            mf_set_baseline = metafeatures[best_baseline_metafeature]
            k_baseline = best_baseline_k
            
            for i in train:
                x_train_baseline.append(mf_set_baseline[i])
                y_train_baseline.append(Y_temp[i])
                
            x_test_baseline= []

            for i in cluster_test:
                x_test_baseline.append(mf_set_baseline[i])

            baseline_classifier = KNeighborsClassifier(n_neighbors=k_baseline)
            baseline_classifier.fit(x_train_baseline,y_train_baseline)

            for i in range(len(cluster_test)):
                if cluster_classifier.predict([x_test_cluster[i]]) == y_test[i]:
                    correct_cluster[cluster] += 1
                    
                if baseline_classifier.predict([x_test_baseline[i]]) == y_test[i]:
                    correct_baseline[cluster] += 1
                    
                total[cluster] += 1

     
        accuracy_cluster.append(sum(correct_cluster)/sum(total))
        accuracy_baseline.append(sum(correct_baseline)/sum(total))

        print("cluster")
        print(correct_cluster)
        print("baseline")
        print(correct_baseline)
        
        file = open("cluster_wise_data.txt","a")
        temp1 = ""
        temp2 = ""
        for i in range(4):
            try:
                temp1 += str(correct_cluster[i]/total[i]) + " , "
            except:
                temp1 += "NIL" + ","
            try:
                temp2 += str(correct_baseline[i]/total[i]) + " , "
            except:
                temp2 += "NIL" + ","

        temp3 = "cluster:" + str(sum(correct_cluster)/sum(total))
        temp4 = "baseline:" + str(sum(correct_baseline)/sum(total))
           
        file.write(temp1[:-1] + "\n" + temp2[:-1] + "\n"+ temp3 + "\n"+ temp4 + "\n")
        file.close()
        print("baseline")
        print(str(sum(accuracy_baseline)))
        print("cluster")
        print(str(sum(accuracy_cluster)))

file = open("hybrid_overall_acc.txt","w")
file.write(str(sum(accuracy_baseline)) + "\n" + str(sum(accuracy_cluster)) + "\n" )
file.write(str(accuracy_baseline) + "\n" + str(accuracy_cluster))
file.close()     




