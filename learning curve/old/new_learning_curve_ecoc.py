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
from sklearn import tree
import math, random

import os

home = os.getcwd()

for k in os.listdir(home):
    path = os.path.join(home, k)
    fig_count = 0
    
    if os.path.isdir(path):#inside dataset folder
        print(k)
        path_dataset1 = os.path.join(path,"dataset1")

        #inside dataset1 folder
            
        #learning_curves folder
        path_learning_curves = os.path.join(path_dataset1,"learning_curves")
        os.makedirs(path_learning_curves, exist_ok=True)

        #metadeta file
        op_file = os.path.join(path_learning_curves,"lc_metadata.txt")
        op_file = open(op_file,"w")
 
        #ecoc files
        for k in os.listdir(path_dataset1):
            
            path = os.path.join(path_dataset1, k)

  
            img_count = k[:-4]
            
                  
            if k.startswith('.') or os.path.isdir(path) or k=="metadata.txt":
                pass
            
            else:
                file_name = path
                print(k)
                

                #importing binary dataset

                file = open(file_name,"r")

                op_file.write(k+"\n")


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
                
                print(n)
                    
                class_0 = np.array(class_0,float)
                class_1 = np.array(class_1,float)
                


                #percentage of each class
                no_0,no_1 = len(class_0),len(class_1)
                c0,c1 = no_0,no_1
                total = no_0 + no_1

                imbalance = abs(c0-c1)/(total)
                imbalance = round(imbalance, 3)
                print(c0,c1)
                print(imbalance)

                op_file.write("class_0" + "," + "class_1"+"\n")
                op_file.write(str(no_0)+","+str(no_1)+"\n")
                op_file.write(str(imbalance)+"\n")

                no_0 /= total
                no_1 /= total

                #fixing the test set size
                test_set_size = max(30,0.2*n)
                step = min(100,.05*n)
                step = int((step/n)*100)
           
                n1 = n - test_set_size

                #accuracy list
                accuracy = [[0 for i in range(100)] for j in range(5,101,step)]

                classifier = tree.DecisionTreeClassifier()


                for n0 in range(100):
            
                    
                    test = []
                    
                    c0_copy = np.copy(class_0)
                    c1_copy = np.copy(class_1)

                    test0 = int(no_0 * test_set_size)
                    test1 = int(no_1 * test_set_size)

                    np.random.shuffle(c0_copy)
                    np.random.shuffle(c1_copy)

                    for x in range(test0):
                        test.append(c0_copy[x])
                    for x in range(test1):
                        test.append(c1_copy[x])

               
                    
                    c0_copy = c0_copy[test0:]
                    c1_copy = c1_copy[test1:]

                    sample_size = 5

                    while sample_size <= 100:
                        train = []

                        train0 = int(no_0 * (sample_size/100) * n1)
                        train1 = int(no_1 * (sample_size/100) * n1)

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
                       
                        accuracy[(sample_size//5)-1][n0] = classifier.score(test_set_x,test_set_y)
              
                        sample_size += step
       

                for i in accuracy:
                    op_file.write(str(i)+",")
                op_file.write("\n")
                    
                accuracy_mean = []
                
                for i in accuracy:
                    accuracy_mean.append(sum(i))
                                         
                
                plt.figure(fig_count)
                plt.title("LC" + str(c0) + "|" + str(c1) + "|" + str(imbalance))
                plt.xlabel("Training examples")
                plt.ylabel("Accuracy")
                plt.ylim((min(accuracy_mean)-5,101))
                plt.xlim((0,101))

                train_sizes = [i for i in range(5,101,step)]
                plt.plot(train_sizes, accuracy_mean, 'o-', color="r",label="Training score")

                path1 = os.path.join(path_learning_curves,"0lc_"+str(img_count)+".png")

                plt.savefig(path1)

                fig_count += 1
                
                plt.figure(fig_count)
                plt.title("LC" + str(c0) + "|" + str(c1) + "|" + str(imbalance))
                plt.xlabel("Training examples")
                plt.ylabel("Accuracy")
                plt.ylim((0,101))
                plt.xlim((0,101))

                train_sizes = [i for i in range(5,101,step)]
                plt.plot(train_sizes, accuracy_mean, 'o-', color="r",label="Training score")

                path1 = os.path.join(path_learning_curves,"1lc_"+str(img_count)+".png")

                plt.savefig(path1)

                fig_count+=1
          

op_file.close()


