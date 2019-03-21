measures = {"overlapping.F1":[0 for i in range(1000)],
            "overlapping.F1v":[0 for i in range(1000)],
            "overlapping.F2":[0 for i in range(1000)],
            "overlapping.F3": [0 for i in range(1000)],
            "overlapping.F4":[0 for i in range(1000)],
            "neighborhood.N1":[0 for i in range(1000)],
            "neighborhood.N2":[0 for i in range(1000)],
            "neighborhood.N3":[0 for i in range(1000)],
            "neighborhood.N4":[0 for i in range(1000)],
            "neighborhood.T1":[0 for i in range(1000)],
            "neighborhood.LSCAvg":[0 for i in range(1000)],
            "linearity.L1":[0 for i in range(1000)],
            "linearity.L2":[0 for i in range(1000)],
            "linearity.L3":[0 for i in range(1000)],
            "dimensionality.T2":[0 for i in range(1000)],
            "dimensionality.T3":[0 for i in range(1000)],
            "dimensionality.T4":[0 for i in range(1000)],
            "balance.C1":[0 for i in range(1000)],
            "balance.C2":[0 for i in range(1000)],
            "network.Density":[0 for i in range(1000)],
            "network.ClsCoef":[0 for i in range(1000)],
            "network.Hubs":[0 for i in range(1000)]}

file = open("complexity.txt").read().split("\n")
count = 0
for i in file:
    if i =='"x"':
        continue
    if count == 23:
        count = 0
    i = i.split(",")
    if count == 0:
        
        temp = i[1].split("\"")
        index = int(temp[1].strip(".txt"))
        count+=1
    else:
        temp = i[0].split("\"")
        value = float(i[1])
        measure = measures[temp[1]]
        measure[index] = value
        count += 1
        
for i in measures.keys():
    file_name = str(i) + ".txt"
    file = open(file_name ,"w")
    count = 0
    for j in measures[i]:
        file.write(str(count) + "," + str(j) + "\n")
        count +=1
    file.close()
    
            
            
            
            
            
            
            
            
