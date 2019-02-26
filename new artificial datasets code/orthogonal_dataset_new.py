from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import operator

home = os.getcwd()

path_dataset_folder = os.path.join(home,"artificial datasets") #artificial dataset folder
os.makedirs(path_dataset_folder, exist_ok=True)

path_ortho_folder = os.path.join(path_dataset_folder,"orthogonal_datasets")
os.makedirs(path_ortho_folder, exist_ok=True)

path_ortho_img_folder = os.path.join(path_dataset_folder,"orthogonal_img_datasets")
os.makedirs(path_ortho_img_folder, exist_ok=True)

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

for k in range(1,21):

    for d in range(10):
        print(dataset_count)

        label = [None for i in range(500)]

        slopes = []

        x = []
        y = []

        for i in range(k):
            n = random.randint(10,70)/100
            x.sort()
            if i > 0:
                while abs(x[i-1] - n) <= 0.2 or abs(x[i-1] - n) > 0.5:
                    n = random.randint(10,70)/100
            x.append(n)
            
        for i in range(k):
            n = random.randint(10,70)/100
            y.sort()
            if i > 0:
                while abs(y[i-1] - n) <= 0.2 or abs(y[i-1] - n) > 0.5:
                    n = random.randint(10,70)/100
            y.append(n)

        x.sort()
        y.sort()
        
        x.append(0.8)
        y.append(1)

        #x = [0.12, 0.35, 0.6, 0.8]
        #y = [0.15, 0.19, 0.54, 1]

        #print(x,y)
        
        temp = 0
        for x1 in x:
            for y1 in y:
                for i in range(500):
                    if X[i] <= x1 and Y[i] <= y1 and label[i] == None:
                        label[i] = temp
                if temp:
                    temp = 0
                else:
                    temp = 1
            if k%2 != 0:
                if temp:
                    temp = 0
                else:
                    temp = 1

            
                    
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
        plt.title("Dataset | " + str(k) + "|" + str(imbalance))


        path_ortho_img = os.path.join(path_ortho_img_folder,str(dataset_count)+".png")   
        plt.savefig(path_ortho_img)
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        path_dataset_file = os.path.join(path_ortho_folder,str(dataset_count)+".txt")
        op_file = open(path_dataset_file,"w")
        
        for i in range(500):
            temp = str(X[i]) +"," + str(Y[i]) + "," + str(float(label[i]))
            op_file.write(temp)
            op_file.write("\n")

        op_file.close()
        
        dataset_count += 1



                
            
            







