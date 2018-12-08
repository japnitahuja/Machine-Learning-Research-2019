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

path_linear_folder = os.path.join(path_dataset_folder,"linear_centre_datasets")
os.makedirs(path_linear_folder, exist_ok=True)

path_linear_img_folder = os.path.join(path_dataset_folder,"linear_centre_img_datasets")
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

centre_x = (min(X) + max(X))/2
centre_y = (min(Y) + max(Y))/2


for p in range(2,22):

    print(p)
    down = p//2
    above = p - down
    

    for k in range(10):
        
        label = [None for i in range(500)]

        slopes = []

        intercepts = {}
        flag = True
        for i in range(above):
            m = random.randint(10,99) / 10
            if flag:
                slopes.append(-1*m)
                flag = False
            else:
                slopes.append(m)
                flag = True
        
        for i in range(down):
            m = random.randint(10,99) / 100
            if flag:
                slopes.append(-1*m)
                flag = False
            else:
                slopes.append(m)
                flag = True

        #slopes = [-4.6, -0.57, 0.51, 6.5, 1000000]
      
        for i in slopes:
            b = centre_y - (i*centre_x)
            intercepts[i] = b

        slopes.sort()
        #slopes.append(1000000)
        intercepts[1000000] = 0

    
        #print(slopes)
        
        temp1 = 0
        for i in slopes:
            for j in range(500):
                        
                    #first quadrant
                    if Y[j] <= i * X[j] + intercepts[i] and label[j] == None  and X[j] >= centre_x and Y[j] >= centre_y:
                        label[j] = temp1

                    #second quadrant
                    if Y[j] <= i * X[j] + intercepts[i] and label[j] == None  and X[j] >= centre_x and Y[j] <= centre_y:
                        label[j] = temp1
                    
                    #third quadrant
                    if Y[j] >= i * X[j] + intercepts[i] and label[j] == None  and X[j] <= centre_x and Y[j] <= centre_y:
                        label[j] = temp1

                     #fourth quadrant
                    if Y[j] >= i * X[j] + intercepts[i] and label[j] == None  and X[j] <= centre_x and Y[j] >= centre_y:
                        label[j] = temp1
                    
            if temp1:
                temp1 = 0
            else:
                temp1 = 1


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
                label[i] = temp1
                if temp1:
                    plt.plot(X[i],Y[i],'go')
                else:
                    plt.plot(X[i],Y[i],'ro')
                count_0 += 1



        imbalance = abs(count_0 - count_1)/500
        plt.title("Dataset | "+ str(p) + " | "+ str(imbalance))

        """
        def abline(slope,intercept):
            axes = plt.gca()
            x_vals = np.array([i/10 for i in range(0,9)])
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals,y_vals,'--')

        
        for i in slopes:
            abline(i,intercepts[i])
        """
        
        path_linear_img = os.path.join(path_linear_img_folder,str(dataset_count)+".png")   
        plt.savefig(path_linear_img)
        #plt.show()
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

       

                
            
            







