import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import os, csv
from scipy import stats
from sklearn import preprocessing
Debug = 0

complexity_features = open("complexity_measures.txt").read().split("\n")
normal_features = open("Classical-Decision_Tree.txt").read().split("\n")
labels = open("nb_vs_qda.txt").read().split("\n")

file = open("cluster_wise_data.txt","w")
file.close()

sum_cluster_wise_acc = [0 for i in range(4)]
count_cluster_wise_superior = [0 for i in range(4)]


def t_test(a,b,c,flag): #greatest in 3 arrays using t test
    t_test = stats.ttest_rel(a,b)

    t_test_value = t_test[0]
    p_value = t_test[1]
    p_value_comparison = 0.05

    #accept the null hypothesis: no difference therefore draw
    if p_value > p_value_comparison:
        if sum(a) > sum(b):
            chosen = a
        else:
            chosen = b
        
    elif t_test_value < 0:
        chosen = b

    else:
        chosen = a

    final = chosen
    if flag:
        t_test = stats.ttest_rel(chosen,c)

        t_test_value = t_test[0]
        p_value = t_test[1]
        p_value_comparison = 0.05

        #accept the null hypothesis: no difference therefore draw
        if p_value > p_value_comparison:
            if sum(chosen) > sum(c):
                final = chosen
            else:
                final = c
            
        elif t_test_value < 0:
            final = c

        else:
            final = chosen

    if final == a:
        return 0
    elif final == b:
        return 1
    elif final == c:
        return 2
    else:
        print("error in t test")
        return -1
    


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

        #save the training set class labels
        train_class_label_file = "train_class_label.txt"
        with open(train_class_label_file, "a") as f:
            writer = csv.writer(f)
            cluster_label = []
            cluster_label.append(n0)
            cluster_label.append(test_fold)
            writer.writerow(cluster_label)
            class_label = []
            for i in test:
                class_label.append(Y_temp[i])
            writer.writerow(class_label)

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

        """
        this is clustering
        """
        #CLUSTER BASED MODEL
        #clustering the training set
        flag = True
        
        while flag:
            k_mean = 4
            kmeans = KMeans(n_clusters = k_mean,n_init=1000, max_iter=10000).fit(raw_data) 
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            SSE = kmeans.inertia_
            flag = False
            
            count = [0,0,0,0]
            for i in labels:
                count[i] += 1
            for i in count:
                if i < 10:
                    flag = True
        """
        end
        """

        #save the centroids
        with open("centroids.txt", "a") as f:
            writer = csv.writer(f)
            row = []
            row.append(n0)
            row.append(test_fold)
            writer.writerow(row)
            writer.writerows(centroids)

        #save the SSE
        with open("SSE.txt", "a") as f:
            writer = csv.writer(f)
            row = []
            row.append(n0)
            row.append(test_fold)
            row.append(SSE)
            writer.writerow(row)

        best_cluster_metafeature = [0,0,0,0]
        best_cluster_k = [0,0,0,0]
        best_cluster_accuracy = [[] for i in range(4)]

        print("clusterbased")
        #internal 10cv
        cluster_count = 0
        for cluster in range(k_mean):
            cluster_folds = [[] for i in range(10)]
            
            X1, X2 , X3, Y = [], [], [], []

            count = 0
            train_cluster_temp = []
            for i in train:
                if labels[count] == cluster:
                    train_cluster_temp.append(i)
                    X1.append(X1_temp[i])
                    X2.append(X2_temp[i])
                    X3.append(X3_temp[i])
                    Y.append(Y_temp[i])
                count += 1
            
            '''       
            #count the number of instances in each class in each cluster
            count_class = [0 for i in range(3)]
            for i in Y:
                count_class[i] += 1
            with open("no_instances_per_class_per_cluster.txt", "a") as f:
                writer = csv.writer(f)
                row = []
                row.append(n0)
                row.append(test_fold)
                row.append(cluster_count)
                cluster_count += 1
                writer.writerow(row)
                writer.writerow(count_class)
            '''
            
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


            #t-test
            accuracy_mean_1 = [[] for i in range(13)]
            accuracy_mean_2 = [[] for i in range(13)]
            accuracy_mean_3 = [[] for i in range(13)]

            if Debug:
                print("T-test")
                print(len(accuracy_mean_1[5]))
                
            for i in range(13):
                for j in range(0,len(accuracy1[i]),10):
                    accuracy_mean_1[i].append(sum(accuracy1[i][j:j+10]))
                    accuracy_mean_2[i].append(sum(accuracy2[i][j:j+10]))
                    accuracy_mean_3[i].append(sum(accuracy3[i][j:j+10]))

            if Debug:
                print(len(accuracy_mean_1[5]))

            best_metafeature_all_k = [None for i in range(13)]

            for i in range(len(accuracy_mean_1)):
                if len(accuracy_mean_1[i]) != 10:
                    continue

                best_metafeature_all_k[i] = t_test(accuracy_mean_1[i],accuracy_mean_2[i],accuracy_mean_3[i],True)

            if Debug:
                print(best_metafeature_all_k)

            accuracies = [accuracy_mean_1,accuracy_mean_2,accuracy_mean_3]
            maximum_acc = None
            best_k = None
            best_metafeature = None
            for i in range(len(best_metafeature_all_k)):
                if best_metafeature_all_k[i] == None:
                    continue
                if maximum_acc == None:
                    best_k = i
                    best_metafeature = best_metafeature_all_k[i]
                    maximum_acc = accuracies[best_metafeature]
                    maximum_acc = maximum_acc[i]
                    
                else:
                    comparison_k = i
                    comparison_metafeature = best_metafeature_all_k[i]
                    comparison_acc = accuracies[comparison_metafeature]
                    comparison_acc = comparison_acc[i]
                    result_ttest = t_test(maximum_acc,comparison_acc,[],False)

                    if Debug:
                        print("Best")
                        print(best_k)
                        print(best_metafeature)
                        print("Comparison")
                        print(comparison_k)
                        print(comparison_metafeature)
                        print("Result")
                        print(sum(maximum_acc),sum(comparison_acc),result_ttest)

                    if result_ttest == 1:
                        best_k = comparison_k
                        best_metafeature = comparison_metafeature
                        maximum_acc = comparison_acc

            best_cluster_metafeature[cluster] = best_metafeature
            best_cluster_k[cluster] = best_k

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

        accuracy_mean_1 = [[] for i in range(13)]
        accuracy_mean_2 = [[] for i in range(13)]
        accuracy_mean_3 = [[] for i in range(13)]
        
        if Debug:
            print("T-test")
            print(len(accuracy_mean_1[5]))
            
        for i in range(13):
            for j in range(0,len(accuracy1[i]),10):
                accuracy_mean_1[i].append(sum(accuracy1[i][j:j+10]))
                accuracy_mean_2[i].append(sum(accuracy2[i][j:j+10]))
                accuracy_mean_3[i].append(sum(accuracy3[i][j:j+10]))

        if Debug:
            print(len(accuracy_mean_1[5]))

        best_metafeature_all_k = [None for i in range(13)]

        for i in range(len(accuracy_mean_1)):
            if len(accuracy_mean_1[i]) != 10:
                continue

            best_metafeature_all_k[i] = t_test(accuracy_mean_1[i],accuracy_mean_2[i],accuracy_mean_3[i],True)

        if Debug:
            print(best_metafeature_all_k)

        accuracies = [accuracy_mean_1,accuracy_mean_2,accuracy_mean_3]
        maximum_acc = None
        best_k = None
        best_metafeature = None
        for i in range(len(best_metafeature_all_k)):
            if best_metafeature_all_k[i] == None:
                continue
            if maximum_acc == None:
                best_k = i
                best_metafeature = best_metafeature_all_k[i]
                maximum_acc = accuracies[best_metafeature]
                maximum_acc = maximum_acc[i]
                
            else:
                comparison_k = i
                comparison_metafeature = best_metafeature_all_k[i]
                comparison_acc = accuracies[comparison_metafeature]
                comparison_acc = comparison_acc[i]
                result_ttest = t_test(maximum_acc,comparison_acc,[],False)

                if Debug:
                    print("Best")
                    print(best_k)
                    print(best_metafeature)
                    print("Comparison")
                    print(comparison_k)
                    print(comparison_metafeature)
                    print("Result")
                    print(sum(maximum_acc),sum(comparison_acc),result_ttest)

                if result_ttest == 1:
                    best_k = comparison_k
                    best_metafeature = comparison_metafeature
                    maximum_acc = comparison_acc

            best_baseline_metafeature = best_metafeature
            best_baseline_k = best_k

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

        #save the train instance and class labels
        for i in range(4):
            with open("train_instance_label.txt", "a") as f:
                writer = csv.writer(f)
                row = []
                row.append(n0)
                row.append(test_fold)
                row.append(i)
                writer.writerow(row)
                writer.writerow(cluster_wise_train[i])

            #save the test instance and class labels
            with open("test_instance_label.txt", "a") as f:
                writer = csv.writer(f)
                row = []
                row.append(n0)
                row.append(test_fold)
                row.append(i)
                writer.writerow(row)
                writer.writerow(cluster_wise_test[i])

        metafeatures = [X1_temp,X2_temp,X3_temp]
        correct_cluster = [0,0,0,0]
        correct_baseline = [0,0,0,0]
        total = [0,0,0,0]

        
        temp = []
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
            
            temp.append(correct_cluster[cluster] / total[cluster])

            cluster_wise_acc_file = "cluster_wise_accuracy.txt"
            with open(cluster_wise_acc_file,"a") as f:
                writer = csv.writer(f)
                row = []
                row.append(n0)
                row.append(test_fold)
                row.append(cluster)
                row.append(correct_cluster[cluster] / total[cluster])
                writer.writerow(row)

        with open("best_cluster_metafeature_cluster.txt", "a") as f:
            writer = csv.writer(f)
            writer.writerow(best_cluster_metafeature)

        with open("best_cluster_metafeature_baseline.txt", "a") as f:
            writer = csv.writer(f)
            temp_best_baseline = []
            temp_best_baseline.append(best_baseline_metafeature)
            writer.writerow(temp_best_baseline)


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

        print("temp")
        print(temp)
        cluster_size = []
        for cluster in cluster_wise_test:
            cluster_size.append(len(cluster))
        for i in range(len(cluster_size)):
            size_temp = 0
            size_index_temp = i
            for j in range(i+1,len(cluster_size)):
                if cluster_size[j] > size_temp:
                    size_temp = cluster_size[j]
                    size_index_temp = j
            temp_t = temp[size_index_temp]
            temp[size_index_temp] = temp[i]
            temp[i] = temp_t
            temp_t = cluster_size[size_index_temp]
            cluster_size[size_index_temp] = cluster_size[i]
            cluster_size[i] = temp_t
        for i in range(len(cluster_size)):
            sum_cluster_wise_acc[i] += temp[i]
            if temp[i] > (sum(correct_baseline)/sum(total)):
                count_cluster_wise_superior[i] += 1

file = open("hybrid_overall_acc.txt","w")
file.write(str(sum(accuracy_baseline)) + "\n" + str(sum(accuracy_cluster)) + "\n" )
file.write(str(accuracy_baseline) + "\n" + str(accuracy_cluster))
file.close()     

for i in range(len(sum_cluster_wise_acc)):
    sum_cluster_wise_acc[i] /= 100

with open("cluster_wise_acc_aggregate.txt", "w") as f:
    writer = csv.writer(f)
    writer.writerow(sum_cluster_wise_acc)

with open("count_cluster_wise_superior.txt", "w") as f:
    writer = csv.writer(f)
    writer.writerow(count_cluster_wise_superior)



