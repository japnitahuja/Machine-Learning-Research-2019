from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import operator

home = os.getcwd()

path_dataset_folder = os.path.join(home,"artificial datasets") #artificial dataset folder
os.makedirs(path_dataset_folder, exist_ok=True)

path_knn_folder = os.path.join(path_dataset_folder,"KNN_datasets")
os.makedirs(path_knn_folder, exist_ok=True)

path_knn_img_folder = os.path.join(path_dataset_folder,"KNN_img_datasets")
os.makedirs(path_knn_img_folder, exist_ok=True)

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

for k_points in range(20,60):

    for n0 in range(5):
        print(dataset_count)
        counter = 0

        train = []
        label = [ None for i in range(500)]
        
        #labelling random points class 0
        while counter != k_points//2:
            rand_int = random.randint(0,249)
            if label[rand_int] == None:
                label[rand_int] = 0
                train.append([X[rand_int],Y[rand_int],0])
                
                counter += 1
  
        #labelling random points class 1
        while counter != k_points:
            rand_int = random.randint(250,499)
            if label[rand_int] == None:
                label[rand_int] = 1
                train.append([X[rand_int],Y[rand_int],1])
                counter += 1
     
        #distance calculation - euclidean 
        def dist_calculation(data1, data2):
            dist = 0
            for i in range(2):
                v1 = data1[i]
                v2 = data2[i]
                #print(v1,v2)
                dist += pow(float(v1) - float(v2), 2)
            return dist

        #Knn 
        def Knn(data, train):
            k = len(train)
            distances = []
            for i in range(len(train)):
                dist = dist_calculation(data,train[i])
                distances.append((dist,train[i][2]))
            distances.sort(key=operator.itemgetter(0))
            neighbours = [0 for i in range(2)]

            count=[0,0]
            
            for i in range(k):
                try:
                    neighbours[int(distances[i][1])] += 1/distances[i][0]
                except: #super close points -- approximate distance is 0
                    
                    neighbours[int(distances[i][1])] += 10000
                    
            maximum = max(neighbours)

            return neighbours.index(maximum) 
        
        #generating the dataset
        for i in range(500):
            label[i] = Knn([X[i],Y[i]], train)
   

        
        plt.figure()

        count_0 = 0
        count_1 = 0
        
        for i in range(500):
            if label[i] == 1:
                plt.plot(X[i],Y[i],'go')
                count_1 += 1

            elif label[i] == 0:
                plt.plot(X[i],Y[i],'ro')
                count_0 += 1

        imbalance = abs(count_0 - count_1)/500
        plt.title("Dataset | " + str(k_points) + " points | " + str(imbalance))
        

        path_knn_img = os.path.join(path_knn_img_folder,str(dataset_count)+".png")   
        plt.savefig(path_knn_img)
        plt.clf()
        plt.cla()
        plt.close()

        path_dataset_file = os.path.join(path_knn_folder,str(dataset_count)+".txt")
        op_file = open(path_dataset_file,"w")

        for i in range(500):
            temp = str(X[i]) +"," + str(Y[i]) + "," + str(float(label[i]))
            op_file.write(temp)
            op_file.write("\n")

        op_file.close()
    
        dataset_count += 1
        
        
                
            
            




