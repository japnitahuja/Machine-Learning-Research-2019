import matplotlib.pyplot as plt
import os
import numpy as np

algo1 = "dt"
algo2 = "lda"

home = os.getcwd()
new_folder = os.path.join(home,"data")
os.makedirs(new_folder, exist_ok=True)

file_acc = open("cluster_wise_accuracy.txt").readlines()
file_complexity = open("complexity_cluster_format_normal.txt").readlines()

file_op = open(os.path.join(os.getcwd(),"data","measures" + ".txt"),"w")

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
    i = i.strip("\n")
    i = i.split(",")
    for j in i:
        baseline_accuracy.append(float(j))


file_op = open(os.path.join(os.getcwd(),"data","Baseline.txt"),"w")
file_op.write("accuracy" + "\n")
file_op.write(str(np.mean(baseline_accuracy)) + "," + str(np.var(baseline_accuracy)) + "," + str(min(baseline_accuracy)) + "," + str(max(baseline_accuracy)))
file_op.write("\n")
file_op.close()


file_es = open(algo1 + "_vs_" + algo2 + ".txt").readlines()
file_train_clusters = open("train_instance_label.txt").readlines()

es = [0 for i in range(800)]

cluster_wise_es = [ [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] for i in range(100)]

for i in file_es:
    i = i.strip("\n")
    i = i.split(",")

    index = int(i[0])

    if index >= 400 and index < 600:
        continue
    elif index >= 600:
        index -= 200

    es[index] = int(i[1])

count = 0
for i in file_train_clusters:
    i = i.strip("\n")
    i = i.split(",")

    if len(i) == 3:
        cluster = int(i[2])
        
    if len(i) > 3:

        for j in i:
            cluster_wise_es[count][cluster][es[int(j)]] += 1


        cluster += 1

    if cluster == 4:
        count +=1
    
for i in cluster_wise_es:
    total = 0
    for j in i:
        total += sum(j)


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

    file_op = open(os.path.join(os.getcwd(),"data","Baseline.txt"),"a")
    file_op.write("complexity measure " + str(measure) + "\n")
    file_op.write(str(np.mean(baseline_measure)) + "," + str(np.var(baseline_measure)) + "," + str(min(baseline_measure)) + "," + str(max(baseline_measure)))
    file_op.write("\n")
    file_op.close()
 
    freq, bins, patches = plt.hist(cluster_measure, bins=10)
    bins_accuracy_count = [0 for i in range(10)]

    bins_accuracy = [[] for i in range(10)]

    bins_total = [0 for i in range(10)]

    bins_es = [[[],[],[]] for i in range(10)]

    bins_mf = [[] for i in range(10)]


    
    for i in range(100):
        for j in range(4):
            measure_cluster = cluster_wise_complexity[i][j][measure]
  
            for k in range(1,11):
                if measure_cluster <= bins[k]:
                    index_bin = k
                    break
            
            bins_total[index_bin-1] += 1

            bins_accuracy[index_bin-1].append(cluster_wise_accuracy[i][j])

            bins_mf[index_bin-1].append(cluster_wise_complexity[i][j][measure])

            for x in range(3):
                bins_es[index_bin-1][x].append(cluster_wise_es[i][j][x])
             
            
                    
    freq, baseline_bins, patches = plt.hist(baseline_measure, bins=10)
    bins_baseline_accuracy = [[] for i in range(10)]
    
    for i in range(100):
        
        measure_baseline = baseline_measure[i]

        for k in range(1,11):
            if measure_baseline <= baseline_bins[k]:
                index_bin = k
                break

        bins_baseline_accuracy[index_bin-1].append(baseline_accuracy[i])


    for i in range(10):
        X = bins_accuracy[i]
        plt.figure()
        freq, bins, p = plt.hist(X, bins=overall_acc_bins)
        axes = plt.gca()
        axes.set_ylim([0,100])
        plt.ylabel("number of datasets")
        plt.xlabel("accuracy")
        plt.title("Complexity measure: " + str(measure))
        path = os.path.join(os.getcwd(),"graphs_acc",str(measure) + "_" + str(i) + ".png")
        plt.savefig(path)
        plt.clf()
        plt.cla()
        plt.close()

    for i in range(10):
        
        X = bins_baseline_accuracy[i]
        plt.figure()
        freq, bins, p = plt.hist(X, bins=overall_acc_bins)
        axes = plt.gca()
        axes.set_ylim([0,100])
        plt.ylabel("number of datasets")
        plt.xlabel("accuracy")
        plt.title("Complexity measure: " + str(measure))
        path = os.path.join(os.getcwd(),"graphs_acc","BASELINE" + str(measure) + "_" + str(i) + ".png")
        plt.savefig(path)
        plt.clf()
        plt.cla()
        plt.close()


    for i in range(10):
        if len(bins_accuracy[i]) != freq[i]:
            break

    file_op = open(os.path.join(os.getcwd(),"data","measures.txt"),"a")

    file_op.write("measure-" + str(measure)+ "\n")

    file_op.write("Bar-Complexity Measure-Accuracy-# of clusters-D-A1-A2" + "\n")
    sep = "-"
    for i in range(10):
        temp = ""
        temp += str(i) + sep
        if len(bins_accuracy[i]) > 0:
            temp += str(round(np.mean(bins_mf[i]),2)) + "," + str(round(np.var(bins_mf[i]),2)) + "," + str(round(min(bins_mf[i]),2)) + "," + str(round(max(bins_mf[i]),2))
            temp += sep
            temp += str(round(np.mean(bins_accuracy[i]),2)) + "," + str(round(np.var(bins_accuracy[i]),2)) + "," + str(round(min(bins_accuracy[i]),2)) + "," + str(round(max(bins_accuracy[i]),2))
            temp += sep
            temp += str(len(bins_accuracy[i]))
            temp += sep
        else:
            temp += ("0" + sep) * 3
            
        for j in range(3):
            if len(bins_es[i][j]) > 1:
                temp += str(np.mean(bins_es[i][j])) + "," + str(np.var(bins_es[i][j])) + "," + str(min(bins_es[i][j])) + "," + str(max(bins_es[i][j]))
                temp += sep
            else:
                temp += "0" + sep
        if i == 0:
            temp += "baseline measure"
            temp += sep 
            temp += str(np.mean(baseline_measure)) + "," + str(np.var(baseline_measure)) + "," + str(min(baseline_measure)) + "," + str(max(baseline_measure))
            temp += sep    
            temp += "baseline accuracy"
            temp += sep
            temp += str(np.mean(baseline_accuracy)) + "," + str(np.var(baseline_accuracy)) + "," + str(min(baseline_accuracy)) + "," + str(max(baseline_accuracy))
            temp += sep
        temp = temp[:-1]
        temp += "\n"
        file_op.write(temp)

    file_op.close()           




    
            


    


        
