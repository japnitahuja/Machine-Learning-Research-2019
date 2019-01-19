#knn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import csv
import numpy as np 

with open("dataset_dt.txt", "r") as f:
    reader = csv.reader(f)
    line = [row for row in reader]

with open("dataset_rf.txt", "r") as f:
    reader = csv.reader(f)
    line3 = [row for row in reader]

line2 = []
with open("t_test_dt_rf.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
       if len(row) != 0:
         line2.append(row)

'''
with open("train_x_0_0.txt","r") as f:
    reader = csv.reader(f)
    line4 = [row for row in reader]

for i in range(len(line)):
    for j in range(len(line[0])):
        if line[i][j] == "":
            print(i)
            print(line[i])

'''
'''
with open("dataset_raw.txt","r") as f:
    reader = csv.reader(f)
    line5 = [row for row in reader]

print(line5[28612])
'''

n = len(line)
m = len(line[0])
#print(n,m)

#print(len(line2))

#X attributes; Y class labels; Z accuracies
X = [[None for i in range(m-1)] for j in range(n)]
#Y = [None for i in range(n)]
Z = [None for i in range(n)]
Z2 = [None for i in range(n)]

Y = []
for row in line2:
    Y.append(int(row[0]))
        
for i in range(n):
    for j in range(m-1):
        X[i][j] = line[i][j]
    Z[i] = line[i][m-1]
    Z2[i] = line3[i][m-1]

#print(len(X),len(Y),len(Z))



#dividing on the basis of class
label_wise = [[] for i in range(m-1)]

for i in range(n):
    index = Y[i]
    label_wise[index].append(i)

print(len(label_wise[0]),len(label_wise[1]),len(label_wise[2]))

accuracy = [[] for i in range(15)]

for k in range(2,3):
    print(k)
    #start of 10 x 10 cv
    for n0 in range(10):

        label_wise_all = []
        for i in label_wise:
            np.random.shuffle(i)
            for j in i:
                label_wise_all.append(j)

        one,two,three,four,five,six,seven,eight,nine,ten = [[] for i in range(10)]
        folds = [one,two,three,four,five,six,seven,eight,nine,ten]

        for j in range(len(label_wise_all)):
            temp = j % 10
            temp = folds[temp]
            temp.append(label_wise_all[j])

        for test_fold in range(10):
            train_x = []
            train_y = []
            train_z = []
            train_z2 = []
            test_x = []
            test_y = []
            test_z = []
            test_z2 = []

            for fold in range(10):
                temp = folds[fold]
                if fold == test_fold:
                    for i in temp:
                        test_x.append(X[i])
                        test_y.append(Y[i])
                        test_z.append(Z[i])
                        test_z2.append(Z2[i])
                else:
                    for i in temp:
                        train_x.append(X[i])
                        train_y.append(Y[i])
                        train_z.append(Z[i])
                        train_z2.append(Z2[i])

            train_y = np.array(train_y).reshape(-1,1)
            test_y = np.array(test_y).reshape(-1,1)
            train_z = np.array(train_z).reshape(-1,1)
            train_z2 = np.array(train_z2).reshape(-1,1)
            test_z = np.array(test_z).reshape(-1,1)
            test_z2 = np.array(test_z2).reshape(-1,1)

            file_name = "train_x_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(train_x)

            file_name = "train_y_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(train_y)

            file_name = "train_dt_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(train_z)

            file_name = "train_rf_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(train_z2)

            file_name = "test_x_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(test_x)

            file_name = "test_y_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(test_y)

            file_name = "test_dt_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(test_z)

            file_name = "test_rf_" + str(n0) + '_' + str(test_fold) + ".txt"
            with open(file_name,"w") as f:
                writer = csv.writer(f)
                writer.writerows(test_z2)

               
