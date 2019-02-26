from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import operator
import itertools

home = os.getcwd()

path_dataset_folder = os.path.join(home,"artificial datasets") #artificial dataset folder
os.makedirs(path_dataset_folder, exist_ok=True)

path_linear_folder = os.path.join(path_dataset_folder,"linear_corner_datasets")
os.makedirs(path_linear_folder, exist_ok=True)

path_linear_img_folder = os.path.join(path_dataset_folder,"linear_corner_img_datasets")
os.makedirs(path_linear_img_folder, exist_ok=True)

dataset_count = 0

#generating a grid of 25 x 25 points
points = []
for i in range(25):
    for j in range(25):
        points.append([i,j])

#scaling points to [0,1]
min_max_scaler = preprocessing.MinMaxScaler()
points = min_max_scaler.fit_transform(points)

#print(points)

X = []
Y = []

for point in points:
    X.append(round(point[0],3))
    Y.append(round(point[1],3))


for m in range(1,21):

    print(m)
    
    down = m//2
    above = m - down

    #print(above,down)
    

    for k in range(10):
        
        label = [None for i in range(500)]

        slopes = []

        for i in range(above):
            slopes.append(random.randint(10,99) / 10)

        for i in range(down):
            slopes.append(random.randint(10,99) / 100)

        slopes.sort()
        slopes.append(1000000)

        #print(slopes)


        temp = 0
        for i in slopes:
            for j in range(500):
                if Y[j] <= i * X[j] and label[j] == None:
                    label[j] = temp
            if temp:
                temp = 0
            else:
                temp = 1

        if temp:
            temp = 0
        else:
            temp = 1

        for j in range(500):
            if label[j] == None:
                label[j] = temp
    

        count_1 = 0
        count_0 = 0

        for i in range(500):
            if label[i] == 1:
                plt.plot(X[i],Y[i],'go')
                count_1 += 1

            elif label[i] == 0:
                plt.plot(X[i],Y[i],'ro')
                count_0 += 1

            elif label[i] == None:
                plt.plot(X[i],Y[i],'bo')
                count_0 += 1



        imbalance = abs(count_0 - count_1)/500
        plt.title("Dataset | "+ str(m) + " | "+ str(imbalance))


        path_linear_img = os.path.join(path_linear_img_folder,str(dataset_count)+".png")   
        plt.savefig(path_linear_img)
        plt.clf()
        plt.cla()
        plt.close()

        path_dataset_file = os.path.join(path_linear_folder,str(dataset_count)+".txt")
        op_file = open(path_dataset_file,"w")

        for i in range(500):
            temp = str(X[i]) +"," + str(Y[i]) + "," + str(float(label[i]))
            op_file.write(temp)
            op_file.write("\n")

        op_file.close()

        dataset_count += 1

       

                
            
            







