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
import operator
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math

import os

home = os.getcwd()
path = os.path.join(home,"learning_curves")
os.makedirs(path, exist_ok=True)
path = os.path.join(home,"lc_metadata.txt")
op_file = open("lc_metadata.txt","w")
 
for i in os.listdir(home):
    path = os.path.join(home, i)
          
    if i.startswith('.') or os.path.isdir(path):
        pass
    else:
        file_name = i
        print(i)

    #importing binary dataset

    file = open(file_name,"r")



    op_file.write(file_name+"\n")


    class_0, class_1 = [],[]
    count = 0
    for i in file:
        i = i.strip("\n")
        i = i.split(",")
        attr = len(i) - 1
        
        if int(i[attr]) ==1:
            class_1.append(i)
        else:
            class_0.append(i)
        count += 1

    print(count)
        
    file.close()

    if count >= 6000:
                
        class_0 = np.array(class_0,float)
        class_1 = np.array(class_1,float)


        #percentage of each class
        no_0,no_1 = len(class_0),len(class_1)

        op_file.write("class_0" + "," + "class_1"+"\n")
        op_file.write(str(no_0)+","+str(no_1)+"\n")

        total = no_0 + no_1
        no_0 /= total
        no_1 /= total

        #sample size of the dataset
        sample_size = math.ceil((300/total)*100)
        start = int((300/total)*100)
        accuracy_mean = []

        classifier = RandomForestClassifier(max_depth=1, random_state=98)

        while sample_size != 101:
            accuracy = 0
            
            print(sample_size)

            sub_sample_size = math.ceil((sample_size * total)/100)
            
            c0,c1 = math.ceil(no_0 * sub_sample_size),math.ceil(no_1 * sub_sample_size)

            if c0 > len(class_0):
                while c0 != len(class_0):
                    c0 -= 1
            if c1 > len(class_1):
                while c1 != len(class_1):
                    c1 -= 1

            print(c0,c1)

            sample = []
            
            np.random.shuffle(class_0)
            np.random.shuffle(class_1)
            
            for i in range(c0):
                sample.append(class_0[i])
            for i in range(c1):
                sample.append(class_1[i])
            
            one,two,three,four,five,six,seven,eight,nine,ten = [[] for i in range(10)]
            folds = [one,two,three,four,five,six,seven,eight,nine,ten]
            

            for i in range(len(sample)):#dividing into folds
                temp = i%10
                temp = folds[temp]
                temp.append(sample[i])

            for x in range(10):
                train = []
                test = []

                for fold in range(10):
                    temp = folds[fold]
                    if fold == x:
                        for i in temp:
                            test.append(i)
                    else:
                        for i in temp:
                            train.append(i)

                print("Train: " + str(len(train)) + "/"+ str(total))
                print("Test: " + str(len(test)) + "/"+ str(total))
            
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
                
                count = 0
                for k in range(len(test_set_x)):
                    prediction = classifier.predict([test_set_x[k]])
                    if prediction == test_set_y[k]:
                        count += 1
                accuracy += (count/len(test_set_x))
            
            accuracy = (accuracy/10)*100

            sample_size += 1
                
            accuracy_mean.append(accuracy)

        for i in accuracy_mean:
            op_file.write(str(i)+",")
        op_file.write("\n")
            
        print(accuracy_mean)

        plt.figure()
        plt.title("Learning curve " + str(c0) + "|" + str(c1) )
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.ylim((0,101))
        plt.xlim((0,101))
        train_sizes = [i for i in range(start,start+len(accuracy_mean))]
        plt.plot(train_sizes, accuracy_mean, 'o-', color="r",label="Training score")
        plt.savefig(learning_curves/filename+"_lc")

        op_file.close()

    else:
        print("less than 6000")
