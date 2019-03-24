from scipy import stats
import csv
import numpy as np 
flag  = 1 #flag = 0 -> 400 vs 400; flag = 1 -> 10 vs 10

with open("hybrid_overall_acc.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

baseline_accuracy = []
for i in range(len(line[2])):
	for j in range(4):
		baseline_accuracy.append(line[2][i])
baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)

with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

cluster_wise_accuracy = []
for i in range(400):
	cluster_wise_accuracy.append(line[i][3])
cluster_wise_accuracy = np.array(cluster_wise_accuracy).astype(np.float64)

if flag == 1:
	temp_baseline = baseline_accuracy[:]
	temp_cluster = cluster_wise_accuracy[:]
	baseline_accuracy = [0 for i in range(10)]
	cluster_wise_accuracy = [0 for i in range(10)]
	for i in range(10):
		for j in range(40):
			baseline_accuracy[i] += temp_baseline[i*10+j]
			cluster_wise_accuracy[i] += temp_cluster[i*10+j]
		baseline_accuracy[i] /= 40
		cluster_wise_accuracy[i] /= 40
	print("len(baseline_accuracy):", len(baseline_accuracy))
	print("len(cluster_wise_accuracy):", len(cluster_wise_accuracy))

t_test = stats.ttest_rel(baseline_accuracy, cluster_wise_accuracy)

print("t test value:", t_test[0])
print("p value:", t_test[1])