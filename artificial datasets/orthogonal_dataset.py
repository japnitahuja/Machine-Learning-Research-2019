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

for dataset_count in range(200):
    print(dataset_count)
    
    label = [None for i in range(500)]

    slopes = []
    
    x = []
    y = []

    for i in range(3):
        n = random.randint(10,70)/100
        x.sort()
        if i > 0:
            while abs(x[i-1] - n) < 0.2 or abs(x[i-1] - n) > 0.5 :
                n = random.randint(10,70)/100
        x.append(n)
        
    for i in range(3):
        n = random.randint(10,70)/100
        y.sort()
        if i > 0:
            while abs(y[i-1] - n) < 0.2 or abs(y[i-1] - n) > 0.5:
                n = random.randint(10,70)/100
        y.append(n)

    x.sort()
    y.sort()

    
    #print(x,y)

    for i in range(500):
        #first row
        if X[i] <= x[0] and Y[i] >= y[2]:
            label[i] = 1
            
        elif X[i] >= x[0] and X[i] < x[1] and Y[i] >= y[2]:
            label[i] = 0

        elif X[i] >= x[1] and X[i] < x[2] and Y[i] >= y[2]:
            label[i] = 1
            
        elif X[i] >= x[2] and Y[i] >= y[2]:
            label[i] = 0

        #second row
        elif X[i] <= x[0] and Y[i] >= y[1]:
            label[i] = 0
        
        elif X[i] >= x[0] and X[i] < x[1] and Y[i] >= y[1]:
            label[i] = 1
       
        elif X[i] >= x[1] and X[i] < x[2] and Y[i] >= y[1]:
            label[i] = 0

        elif X[i] >= x[2] and Y[i] > y[1]:
            label[i] = 1
        
        #third row
        elif X[i] <= x[0] and Y[i] >= y[0]:
            label[i] = 1
        
        elif X[i] >= x[0] and X[i] < x[1] and Y[i] >= y[0]:
            label[i] = 0
       
        elif X[i] >= x[1] and X[i] < x[2] and Y[i] >= y[0]:
            label[i] = 1

        elif X[i] >= x[2] and Y[i] > y[0]:
            label[i] = 0

        #fourth row
        elif X[i] <= x[0] and Y[i] <= y[0]:
            label[i] = 0
        
        elif X[i] >= x[0] and X[i] < x[1] and Y[i] <= y[0]:
            label[i] = 1
       
        elif X[i] >= x[1] and X[i] < x[2] and Y[i] <= y[0]:
            label[i] = 0

        elif X[i] >= x[2] and Y[i] <= y[0]:
            label[i] = 1
          
        
       

 

    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    
    

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
    #plt.title("Dataset | " + str(slopes[0]) + "," + str(slopes[1])+ ","
             # + str(slopes[2])+ "," + str(slopes[3])+ " | " + str(imbalance))


    path_ortho_img = os.path.join(path_ortho_img_folder,str(dataset_count)+".png")   
    plt.savefig(path_ortho_img)
    plt.clf()
    plt.cla()
    plt.close()

    path_dataset_file = os.path.join(path_ortho_folder,str(dataset_count+1)+".txt")
    op_file = open(path_dataset_file,"w")

    for i in range(500):
        temp = str(X[i]) +"," + str(Y[i]) + "," + str(float(label[i]))
        op_file.write(temp)
        op_file.write("\n")

    op_file.close()

   

            
        
        






