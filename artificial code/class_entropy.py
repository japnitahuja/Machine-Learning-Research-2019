from scipy.stats import entropy 
from scipy.stats import pearsonr 
import csv
import matplotlib.pyplot as plt
import numpy as np

#Read cluster Data -- train
cluster_train = []
with open("train_instance_label.txt", "r") as f:
	reader = csv.reader(f)
	cnt = 0
	temp = []
	cnt2 = 0
	for row in reader:
		if cnt % 2 == 1:
			temp.append(row)
			cnt2 += 1
		cnt += 1
		if cnt2 == 4:
			cluster_train.append(temp)
			temp = []
			cnt2 = 0

with open("dt_vs_knn.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

labels = [None for i in range(1000)]
for i in range(len(line)):
	labels[int(line[i][0])] = int(line[i][1])

entropy_baseline = []
entropy_cluster = []
for n0 in range(100):
	temp_baseline = [0, 0, 0]
	cnt_baseline = 0
	for cluster in range(4):
		temp_cluster = [0, 0, 0]
		cnt_cluster = 0
		for i in cluster_train[n0][cluster]:
			i = int(i)
			if labels[i] != None:
				temp_cluster[labels[i]] += 1
				cnt_cluster += 1
		for i in range(3):
			temp_baseline[i] += temp_cluster[i]
			temp_cluster[i] /= cnt_cluster
		cnt_baseline += cnt_cluster
		entropy_cluster.append(entropy(temp_cluster))
	for i in range(3):
		temp_baseline[i]/= cnt_baseline
	entropy_baseline.append(entropy(temp_baseline))

#calculate the difference between baseline and cluster wise entropy
entropy_diff = []
for n0 in range(100):
	for cluster in range(4):
		entropy_diff.append(entropy_baseline[n0]-entropy_cluster[n0*4+cluster])

#get the accuracies
with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_accuracy = [row for row in reader]
	cluster_wise_accuracy = np.array(cluster_wise_accuracy).astype(np.float64)

with open("hybrid_overall_acc.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

baseline_accuracy = line[2][:]
baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)

#calculate difference between baseline accuracy and cluster wise accuracy
acc_diff = []
for n0 in range(100):
	for cluster in range(4):
		acc_diff.append(cluster_wise_accuracy[n0*4+cluster][3] - baseline_accuracy[n0])


plt.scatter(entropy_diff, acc_diff)
plt_name = "class_entropy" + ".png"
plt.title("r^2="+str(pearsonr(entropy_diff,acc_diff)[0]**2))
plt.savefig(plt_name, dpi=300)


