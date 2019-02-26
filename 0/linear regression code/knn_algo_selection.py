#knn 
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np 
import os

home = os.getcwd()
train_dt_path = os.path.join(home,"predicted_dt")
train_rf_path = os.path.join(home,"predicted_rf")
test_dt_path = os.path.join(home,"predicted_dt_test")
test_rf_path = os.path.join(home,"predicted_rf_test")

accuracy = [[] for j in range(13)]

for k in range(2,13):
    for n0 in range(10):
        for fold in range(10):
            #print(n0,fold)
            train_x_mf = "train_x_" + str(n0) + '_' + str(fold) + ".txt"
            x_acc = "pred_" + str(n0) + '_' + str(fold) + ".txt"
            train_y_file = "train_y_" + str(n0) + '_' + str(fold) + ".txt"
            test_x_mf = "test_x_" + str(n0) + '_' + str(fold) + ".txt"
            test_y_file = "test_y_" + str(n0) + '_' + str(fold) + ".txt"

            print(k,train_x_mf)
            
            with open(train_x_mf,"r") as f:
                reader = csv.reader(f)
                train_mf = [row for row in reader]
            with open(train_y_file,"r") as f:
                reader = csv.reader(f)
                train_y = [row for row in reader]
            with open(test_x_mf,"r") as f:
                reader = csv.reader(f)
                test_mf = [row for row in reader]
            with open(test_y_file,"r") as f:
                reader= csv.reader(f)
                test_y = [row for row in reader]
            os.chdir(train_dt_path)
            with open(x_acc,"r") as f:
                reader = csv.reader(f)
                train_dt_acc = [row for row in reader]
            os.chdir(train_rf_path)
            with open(x_acc,"r") as f:
                reader = csv.reader(f)
                train_rf_acc = [row for row in reader]
            os.chdir(test_dt_path)
            #print(os.getcwd())
            with open(x_acc,"r") as f:
                reader = csv.reader(f)
                test_dt_acc = [row for row in reader]
            os.chdir(test_rf_path)
            with open(x_acc,"r") as f:
                reader = csv.reader(f)
                test_rf_acc = [row for row in reader]
            os.chdir(home)

            n = len(train_mf)
            m = len(train_mf[0])

            train_x = []
            for i in range(n):
                row = []
                for j in range(m):
                    #print(train_mf[i][j],i,j)
                    row.append(train_mf[i][j])
                row.append(train_dt_acc[i][0])
                row.append(train_rf_acc[i][0])
                train_x.append(row)
                #print(row)

            train_y = np.array(train_y).reshape(len(train_y))

            n_t = len(test_mf)
            #print(train_x)
            #print(train_y)

            test_x = []
            for i in range(n_t):
                row = []
                for j in range(m):
                    row.append(test_mf[i][j])
                row.append(test_dt_acc[0][i])
                row.append(test_rf_acc[0][i])
                #print(row)
                test_x.append(row)

            train_y = np.array(train_y).astype(np.float64)
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(train_x,train_y)
#            print(classifier.score(train_x,train_y))

            for i in range(len(test_x)):
                for j in range(len(test_x[0])):
                    if test_x[i][j] == "inf":
                        test_x[i][j] = 100000.0
            
            test_y = np.array(test_y).reshape(len(test_y))
            test_y = test_y.astype(np.float64)
            #print(test_y)

            test_x = np.array(test_x)
            test_x = test_x.astype(np.float64)
            #print(test_x)

            acc = classifier.score(test_x,test_y)
            '''
            print(classifier.predict_proba(test_x))
            print(acc)
            print(classifier.predict(test_x))
            print(test_y)
            '''
            accuracy[k].append(acc)

#print(accuracy)

with open("knn_acc.txt","w") as f:
    writer = csv.writer(f)
    writer.writerows(accuracy)



