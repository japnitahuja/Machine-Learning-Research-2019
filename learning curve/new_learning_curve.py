"""
import dataset
subsampling: randomly choose percent of data with atleast 10 instances
stratified: maintain the percentage 
10-folds divide
10x10 fold cross validation

size of dataset- 6000 or ore instances as start is 300 as 30 testcases and 300
is 5% of the whole dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble
import math, random

import os

home = os.getcwd()

path_dataset_folder = os.path.join(home,"main_folder") #main dataset folder
learning_curves = []
if os.path.exists(path_dataset_folder):

    #making learning_curves folder
    path_lc_folder = os.path.join(home,"learning_curves_rf")
    os.makedirs(path_lc_folder, exist_ok=True)

    #already done curves
    for curve in os.listdir(path_lc_folder):
        curve = curve.strip(".png")
        learning_curves.append(curve)

    print(learning_curves)

    #metadeta folder 
    path_metadata = os.path.join(home,"metadata")
    os.makedirs(path_metadata, exist_ok=True)

    #learning curve meta data file
    op_file_path = os.path.join(path_metadata,"learning_curves_rf.txt")
    op_file = open(op_file_path,"a")
    op_file.close()

    #error file
    error_file_path = os.path.join(path_metadata,"error_rf.txt")
    error_file = open(error_file_path,"w")
   
    
    for dataset in os.listdir(path_dataset_folder):

        #check for hidden files
        if dataset.startswith('.'):
            continue

        if dataset.strip(".txt") in learning_curves:
            continue
        
        print(dataset)
        #each dataset file
        path_dataset = os.path.join(path_dataset_folder, dataset)

        #importing binary dataset
        file = open(path_dataset,"r")
        
        
        try:
            #dividing into classes
            class_0, class_1 = [],[]
            n = 0
            for i in file:
                i = i.strip("\n")
                i = i.split(",")
                attr = len(i) - 1
                if len(i)>1:
                    if i[attr] == '1.0':
                        class_1.append(i)
                    else:
                        class_0.append(i)
                    n += 1
            
            print("n:" + str(n))
                
            class_0 = np.array(class_0,float)
            class_1 = np.array(class_1,float)
            
            LC_points = 20
            K = 100
            max_size_step = 100

            #percentage of each class
            no_0,no_1 = len(class_0),len(class_1)
            c0,c1 = no_0,no_1
            total = no_0 + no_1

            imbalance = abs(c0-c1)/(total)
            imbalance = round(imbalance, 3)
            print(c0,c1)
            print(imbalance)

            if (14*total) >= (15 * abs(no_1-no_0)):
                op_file = open(op_file_path,"a")
                #writing to file
                op_file.write(dataset.strip(".txt") + ",")
                no_0_proportion = no_0 / total
                no_1_proportion = no_1 / total
                
                #fixing the test set size
                test_set_size = max(30,int(0.2*n))
                test_0 = int(test_set_size * no_0_proportion)
                test_1 = test_set_size - test_0

                #print("test: " + str(test_set_size))
                #print("test1: " + str(test_1))
                #print("test0: " + str(test_0))

                n1 = n - test_set_size
                
                step = min(max_size_step,int(.05*n1))
                print("step: " + str(step) + "\n")

                train_sizes = []    

                #accuracy list
                accuracy = [[0 for i in range(100)] for j in range(LC_points)]

                #classifier = tree.DecisionTreeClassifier()
                classifier = sklearn.ensemble.RandomForestClassifier(n_estimators = 64)

            
                for n0 in range(K):
            
                    test = []

                    c0_copy = np.copy(class_0)
                    c1_copy = np.copy(class_1)

                    np.random.shuffle(c0_copy)
                    np.random.shuffle(c1_copy)

                    for x in range(test_0):
                        test.append(c0_copy[x])
                    for x in range(test_1):
                        test.append(c1_copy[x])

                    c0_copy = c0_copy[test_0:]
                    c1_copy = c1_copy[test_1:]

                    sample_size = step
                    index = 0
                    
                    while index < LC_points:
                        train = []

                        train0 = int(no_0_proportion * sample_size)
                        train1 = sample_size - train0
                        #train1 = int(no_1_proportion * sample_size)

                        if n0 == 0:
                            train_sizes.append(train0 + train1)

                        if index == 19:
                            train0 = len(c0_copy)
                            train1 = len(c1_copy)

                        np.random.shuffle(c0_copy)
                        np.random.shuffle(c1_copy)
       
                        for x in range(0,train0):
                            train.append(c0_copy[x])
                        for x in range(0,train1):
                            train.append(c1_copy[x])
        
                        train_set_x = []
                        train_set_y = []
                        test_set_x = []
                        test_set_y = []
                        
                        for z in train:
                            train_set_x.append(z[:-1])
                            train_set_y.append(z[-1])
                        for z in test:
                            test_set_x.append(z[:-1])
                            test_set_y.append(z[-1])

                        classifier.fit(train_set_x,train_set_y)
                       
                        accuracy[index][n0] = classifier.score(test_set_x,test_set_y)

                        index +=1
                        sample_size += step
                accuracy_mean = []
                
                for i in accuracy:
                    accuracy_mean.append(sum(i))

                temp = ""
                #writing accuracy to the file
                for i in accuracy_mean:
                    temp += str(i)+","
                op_file.write(temp[:-1]+"\n")

                """
                #writing train sizes to the file
                for i in train_sizes:
                    op_file.write(str(i)+",")
                op_file.write("\n")
                """                                        
                #print(train_sizes)

                #plotting the learning curve
                plt.figure()
              
                plt.title("LC" + str(c0) + "|" + str(c1) + "|" + str(imbalance))
                plt.xlabel("Training size")
                plt.ylabel("Accuracy")
                plt.ylim((min(accuracy_mean)-5,max(accuracy_mean)+5))
                plt.xlim((train_sizes[0]-5,train_sizes[19]+5))

                #train_sizes = [i for i in range(5,101,5)]
                plt.plot(train_sizes, accuracy_mean, 'o-', color="r",label="Training score")

                fig_count = dataset.strip(".txt")
                
                path1 = os.path.join(path_lc_folder,fig_count +".png")

                plt.savefig(path1)
                plt.clf()
                plt.cla()
                plt.close()

                op_file.close()

            else:
                error_file = open(error_file_path,"a")
                error_file.write("Imbalance: " + str(imbalance) + " " + dataset)
                error_file.write("\n")
                error_file.close()

        except:

            error_file = open(error_file_path,"a")
            error_file.write(dataset)
            error_file.write("\n")
            error_file.close()
            







