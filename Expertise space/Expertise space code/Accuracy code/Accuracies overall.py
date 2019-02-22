import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.ensemble
from sklearn import tree
import math, random
from scipy import stats
import os

home = os.getcwd()

algo1 = "dt"
algo2 = "knn"
algo3 = "nb"
algo4  = "rf"
algo5 = "lda"
algo6 = "qda"

path_dataset_folder = os.path.join(home,"main_folder")
path_metadata_folder = os.path.join(home,"metadata")

os.makedirs(path_metadata_folder, exist_ok=True)

op_file1_path = os.path.join(path_metadata_folder,"accuracy_" + algo1 + ".txt")
op_file2_path = os.path.join(path_metadata_folder,"accuracy_" + algo2 + ".txt")
op_file3_path = os.path.join(path_metadata_folder,"accuracy_" + algo3 + ".txt")
op_file4_path = os.path.join(path_metadata_folder,"accuracy_" + algo4 + ".txt")
op_file5_path = os.path.join(path_metadata_folder,"accuracy_" + algo5 + ".txt")
op_file6_path = os.path.join(path_metadata_folder,"accuracy_" + algo6 + ".txt")


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

        
        accuracy1 = [0 for i in range(100)]
        accuracy2 = [0 for i in range(100)]
        accuracy3 = [0 for i in range(100)]
        accuracy4 = [0 for i in range(100)]
        accuracy5 = [0 for i in range(100)]
        accuracy6 = [0 for i in range(100)]

        classifier1 = tree.DecisionTreeClassifier()
        classifier2 = KNeighborsClassifier(n_neighbors=7)
        classifier3 = GaussianNB(var_smoothing=0)
        classifier4 = sklearn.ensemble.RandomForestClassifier(n_estimators=64)
        classifier5 = LinearDiscriminantAnalysis()
        classifier6 = QuadraticDiscriminantAnalysis()
        
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
            accuracy1[n0] = classifier1.score(test_set_x,test_set_y)
            accuracy2[n0] = classifier2.score(test_set_x,test_set_y)
            accuracy3[n0] = classifier3.score(test_set_x,test_set_y)
            accuracy4[n0] = classifier4.score(test_set_x,test_set_y)
            accuracy5[n0] = classifier5.score(test_set_x,test_set_y)
            accuracy6[n0] = classifier6.score(test_set_x,test_set_y)

        files = [op_file1_path,op_file2_path,op_file3_path,op_file4_path,op_file5_path,op_file6_path]
        accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]
        #writing accuracy to the file
        for i in range(6):
            file_op = open(files[i],"a")
            acc = accuracies[i]
            dataset_number = dataset.strip(".txt")
            file_op.write(dataset_number+",")

            temp = ""
            for i in acc:
                temp += (str(i)+",")

            file_op.write(temp[:-1] + "\n")

            file_op.close()
            






