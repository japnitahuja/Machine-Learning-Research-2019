import matplotlib.pyplot as plt
import os

home = os.getcwd()
new_folder = os.path.join(home,"data")
os.makedirs(new_folder, exist_ok=True)

home = os.getcwd()
new_folder = os.path.join(home,"graphs_acc")
os.makedirs(new_folder, exist_ok=True)
new_folder = os.path.join(home,"graphs_measures")
os.makedirs(new_folder, exist_ok=True)

file_op = open(os.path.join(os.getcwd(),"data","cluster_distribution_data.txt") ,"w")
file_acc = open("cluster_wise_accuracy.txt").readlines()
file_complexity = open("complexity_cluster_format_normal.txt").readlines()  

cluster_wise_accuracy = [[0,0,0,0] for i in range(100)]
cluster_wise_complexity = [[0,0,0,0] for i in range(100)]

all_cluster_accuracies = []

outer_index = 0
for i in range(400):
    acc = file_acc[i]
    
    complexity = []
    for j in file_complexity[i].split(","):
        complexity.append(float(j))
    acc = acc.strip()
    acc = acc.split(",")
    inner_index = int(acc[2])

    cluster_wise_accuracy[outer_index][inner_index] = float(acc[3])
    all_cluster_accuracies.append(float(acc[3]))
    cluster_wise_complexity[outer_index][inner_index] = complexity
    if inner_index == 3:
        outer_index += 1

baseline_accuracy = []
baseline_complexity = []

file_acc = open("baseline_accuracy.txt","r")
file_complexity = open("complexity_baseline_format_normal.txt").readlines()

for i in file_complexity:
    i = i.split(",")
    temp = []
    for j in i:
        temp.append(float(j))
    baseline_complexity.append(temp)

for i in file_acc:
    i = i.strip(" ")
    i = i.strip("\n\n")
    i = i.split(",")
    for j in i:
        baseline_accuracy.append(float(j))

freq, overall_acc_bins, patches = plt.hist(all_cluster_accuracies, bins=10)
path = os.path.join(os.getcwd(),"graphs_acc","baseline.png")
plt.savefig(path)
plt.clf()
plt.cla()
plt.close()

for measure in range(22):
    baseline_measure = []
    cluster_measure = []

    for i in range(100):
        for j in range(4):
            cluster_measure.append(cluster_wise_complexity[i][j][measure])
        baseline_measure.append(baseline_complexity[i][measure])

    file_op.write(str(measure)+ "\n\n")
    freq, bins, patches = plt.hist(cluster_measure, bins=10)

    
    plt.figure()
    freq, baseline_bins, patches = plt.hist(baseline_measure, bins=10)
    path = os.path.join(os.getcwd(),"graphs_measures","baseline_" + str(measure) + ".png")
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    file_op.write("baseline bins"+ "\n\n")
    file_op.write(str(baseline_bins)+ "\n\n")
    for i in baseline_bins:
        file_op.write(str(i))
    file_op.write("\n\n")
    file_op.write(str(freq)+ "\n\n")

    plt.figure()
    freq, bins, p = plt.hist(cluster_measure, bins=baseline_bins)
    path = os.path.join(os.getcwd(),"graphs_measures","cluster" + str(measure) + ".png")
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    file_op.write("clusters in the baseline range"+ "\n\n")
    file_op.write(str(freq)+ "\n\n")
    file_op.write(str(len(freq))+ "\n\n")

    plt.figure()
    freq, bins, p = plt.hist(cluster_measure, bins=10)
    path = os.path.join(os.getcwd(),"graphs_measures","cluster_bins_" + str(measure) + ".png")
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()

    file_op.write("clusters" + "\n\n")
    file_op.write(str(bins) + "\n\n")
    for i in bins:
        file_op.write(str(i))
    file_op.write(str(freq) + "\n\n")
    file_op.write("")

    
file_op.close()  
            
        
    
            


    


        
