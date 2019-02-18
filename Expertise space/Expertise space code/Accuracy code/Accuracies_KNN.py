import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math, random
from scipy import stats
import os

home = os.getcwd()

algorithm = "k_nearest_neighbour"

path_dataset_folder = os.path.join(home,"main_folder")
path_metadata_folder = os.path.join(home,"metadata")

os.makedirs(path_metadata_folder, exist_ok=True)

op_file_path = os.path.join(path_metadata_folder,"accuracy_" + algorithm + ".txt")

#clear file
op_file = open(op_file_path,"w")
op_file.close()


if os.path.exists(path_dataset_folder):

    for dataset in os.listdir(path_dataset_folder):

        #check for hidden files
        if dataset.startswith('.'):
            continue
        print(dataset)
        #open dataset
        file = open(os.path.join(path_dataset_folder,dataset), "r")
        
        #dividing into 2 seperate class arrays
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

        class_0 = np.array(class_0,float)
        class_1 = np.array(class_1,float)
                
        #closing dataset
        file.close()
            
        K = 100 #number of repetitions of accuracy values

        #percentage of each class
        no_0,no_1 = len(class_0),len(class_1)
        total = no_0 + no_1

        #print(no_0,no_1)

        no_0_proportion = no_0 / total
        no_1_proportion = no_1 / total

        #fixing the test set size
        test_set_size = max(30,int(0.2*n))
        test_0 = int(test_set_size * no_0_proportion)
        test_1 = test_set_size - test_0

        #accuracy list
        #accuracy_dt = [0 for i in range(100)]
        accuracy = [0 for i in range(100)]

        classifier = KNeighborsClassifier(n_neighbors=7)
        
        for n0 in range(K):

            #test set
            test_set_x = []
            test_set_y = []

            c0_copy = np.copy(class_0)
            c1_copy = np.copy(class_1)

            np.random.shuffle(c0_copy)
            np.random.shuffle(c1_copy)

            #stratification
            for x in range(test_0):
                test_set_x.append(c0_copy[x][:-1])
                test_set_y.append(c0_copy[x][-1])
            for x in range(test_1):
                test_set_x.append(c1_copy[x][:-1])
                test_set_y.append(c1_copy[x][-1])

            #print(len(test_set_x))

            c0_copy = c0_copy[test_0:]
            c1_copy = c1_copy[test_1:]

            #train set
            train_set_x = []
            train_set_y = []
            

            for x in c0_copy:
                train_set_x.append(x[:-1])
                train_set_y.append(x[-1])

            for x in c1_copy:
                train_set_x.append(x[:-1])
                train_set_y.append(x[-1])

            #classifier model generation
            #classifier_dt.fit(train_set_x,train_set_y)
            classifier.fit(train_set_x,train_set_y)

            #accuracies
            #accuracy_dt[n0] = classifier_dt.score(test_set_x,test_set_y)
            accuracy[n0] = classifier.score(test_set_x,test_set_y)

        #writing accuracy to the file
        file_op = open(op_file_path,"a")
        dataset_number = dataset.strip(".txt")
        file_op.write(dataset_number+",")

        temp = ""
        for i in accuracy:
            temp += (str(i)+",")

        file_op.write(temp[:-1] + "\n")

        file_op.close()
            

        """
        #T test
        t_test = stats.ttest_rel(accuracy_dt,accuracy_rf)

        t_test_value = t_test[0]
        p_value = t_test[1]
        p_value_comparison = 0.05

        #accept the null hypothesis: no difference therefore draw
        if p_value > p_value_comparison:
            t_test_value = 0
            
        if t_test_value < 0 and x_coord > y_coord:
            print("ERROR")

        #print(t_test_value,p_value)
        #print(x_coord,y_coord)

        #writing accuracy to the file
        op_file = open(op_file_path,"a")
        dataset_number = dataset.strip(".txt")
        op_file.write(dataset_number + "," + str(x_coord) + "," + str(y_coord)
                      + "," + str(t_test_value) + "\n")
        op_file.close()
        """






