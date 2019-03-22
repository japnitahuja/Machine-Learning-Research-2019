import csv
import numpy as np 

#read linear regression coefficient
with open("coefficient.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]


with open("complexity_cluster_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_complexity = [row for row in reader]
	cluster_wise_complexity = np.array(cluster_wise_complexity).astype(np.float64)

with open("complexity_baseline_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	baseline_complexity = [row for row in reader]
	baseline_complexity = np.array(baseline_complexity).astype(np.float64)

with open("n0_correct_cluster.txt", "r") as f:
	reader = csv.reader(f)
	n0_correct_cluster = [row for row in reader]

with open("n0_correct_baseline.txt", "r") as f:
	reader = csv.reader(f)
	n0_correct_baseline = [row for row in reader]

##calculate difference between baseline complexity and cluster wise complexity
for feature in range(22):
	for n0 in range(100):
		for cluster in range(4):
			cluster_wise_complexity[n0*4+cluster][feature] -= baseline_complexity[n0][feature]

coefficient = line[0]
selected_features = line[1]

cluster_select = []
final_acc = []

for n0 in range(100): 
	no_corr = 0
	no_total = 0
	temp_select = []
	for cluster in range(4):
		x = []
		for feature in selected_features:
			feature = int(feature)
			x.append(cluster_wise_complexity[n0*4+cluster][feature])
		result = 0
		for i in range(len(coefficient)):
			result += float(coefficient[i])*x[n0*4+cluster][i]
		if result > 0:
			no_corr += int(n0_correct_cluster[n0*4+cluster][0])
			temp_select.append("c")
		else:
			no_corr += int(n0_correct_baseline[n0*4+cluster][0])
			temp_select.append("b")
		no_total += int(n0_correct_baseline[n0*4+cluster][1])

	cluster_select.append(temp_select)
	final_acc.append(no_corr/no_total)

with open("final_result.txt", "w") as f:
	writer = csv.writer(f)
	writer.writerow(final_acc)
	writer.writerow("\n")
	writer.writerows(cluster_select)







