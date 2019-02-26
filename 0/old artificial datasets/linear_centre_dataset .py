from sklearn import preprocessing, neighbors
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import operator

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

for dataset_count in range(200):
    print(dataset_count)
    
    label = [None for i in range(500)]

    slopes = []

    #centre
    centre_x = X[250]
    centre_y = Y[12]

    # below y = x
    m2 = random.randint(10,99) / 100
    m3 = random.randint(10,99) / 100
    while abs(m2-m3) <  0.5:
        m2 = random.randint(10,99) / 100
        m3 = random.randint(10,99) / 100
    

    # above y = x
    m0 = random.randint(10,100) / 10
    m1 = random.randint(10,100) / 10
    while abs(m0-m1) <  5:
        m0 = random.randint(10,99) / 10
        m1 = random.randint(10,99) / 10

    m4 =random.randint(10,99) / 100
    m4 = m4*(-1)
   
    slopes.append(m0)
    slopes.append(m1)
    slopes.append(m2)
    slopes.append(m3)
    slopes.sort()

    b0 = centre_y - (m0*centre_x)
    b1 = centre_y - (m1*centre_x)
    b2 = centre_y - (m2*centre_x)
    b3 = centre_y - (m3*centre_x)
    b4 = centre_y - (m4*centre_x)

    
    #print(b0,b1,b2,b3)
    #print(slopes)

   

    intercepts = {m0:b0,m1:b1,m2:b2,m3:b3,m4:b4}

    for i in range(500):
        if ((X[i] * slopes[0]) + intercepts[slopes[0]]) <= Y[i] and ((X[i] * slopes[1]) + intercepts[slopes[1]]) > Y[i] :
            label[i] = 0
        
        elif ((X[i] * slopes[1]) + intercepts[slopes[1]]) <= Y[i] and ((X[i] * slopes[2]) + intercepts[slopes[2]]) > Y[i] :
            label[i] = 1
        
        elif ((X[i] * slopes[2]) + intercepts[slopes[2]]) <= Y[i] and ((X[i] * slopes[3]) + intercepts[slopes[3]]) > Y[i] :
            label[i] = 0
        
        elif ((X[i] * slopes[3]) + intercepts[slopes[3]]) <= Y[i] and ((X[i] * m4) + intercepts[m4]) < Y[i] :
            label[i] = 1
        
        elif ((X[i] * slopes[0]) + intercepts[slopes[0]]) <= Y[i] and ((X[i] * m4) + intercepts[m4]) > Y[i] :
            label[i] = 0
            
        elif ((X[i] * slopes[0]) + intercepts[slopes[0]]) >= Y[i] and ((X[i] * slopes[1]) + intercepts[slopes[1]]) < Y[i] :
            label[i] = 1
        
        elif ((X[i] * slopes[1]) + intercepts[slopes[1]]) >= Y[i] and ((X[i] * slopes[2]) + intercepts[slopes[2]]) < Y[i] :
            label[i] = 0
        
        elif ((X[i] * slopes[2]) + intercepts[slopes[2]]) >= Y[i] and ((X[i] * slopes[3]) + intercepts[slopes[3]]) < Y[i] :
            label[i] = 1

        elif ((X[i] * slopes[3]) + intercepts[slopes[3]]) >= Y[i] and ((X[i] * m4) + intercepts[m4]) > Y[i] :
            label[i] = 0

        elif ((X[i] * slopes[0]) + intercepts[slopes[0]]) >= Y[i] and ((X[i] * m4) + intercepts[m4]) < Y[i] :
            label[i] = 1
        
        else:
            label[i] = 0

      
    count_1 = 0
    count_0 = 0

    for i in range(500):
        if label[i] == 1:
            plt.plot(X[i],Y[i],'go')
            count_1 += 1

        elif label[i] == 0:
            plt.plot(X[i],Y[i],'ro')
            count_0 += 1

        elif label[i] == 2:
            plt.plot(X[i],Y[i],'bo')
            count_0 += 1

        

    imbalance = abs(count_0 - count_1)/500
    plt.title("Dataset | " + str(slopes[0]) + "," + str(slopes[1])+ ","
              + str(slopes[2])+ "," + str(slopes[3])+ " | " + str(imbalance))

    def abline(slope, intercept):
    
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')


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

   

            
        
        






