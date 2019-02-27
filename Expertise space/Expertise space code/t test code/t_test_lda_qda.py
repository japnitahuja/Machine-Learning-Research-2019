import csv
from scipy import stats

algo1_name = "lda"
algo2_name = "qda"

algo1_file_name = "accuracy_" + algo1_name + ".txt"
algo2_file_name = "accuracy_" + algo2_name + ".txt"

#open the files of accuracy for two algorithms
with open(algo1_file_name, "r") as f:
	reader = csv.reader(f)
	algo1 = [row for row in reader]
with open(algo2_file_name, "r") as f:
	reader = csv.reader(f)
	algo2 = [row for row in reader]

n = len(algo1)

#number of datasets
n_d = 30000

label = ["NULL" for i in range(n_d)]

for i in range(n):
	print(i)
	algo1_temp = []
	algo2_temp = []
	t_test = []
	for j in range(1,len(algo1[0])):
		algo1_temp.append(float(algo1[i][j]))
		algo2_temp.append(float(algo2[i][j]))
	#print(algo1_temp)
	#print(algo2_temp)
	t_test = stats.ttest_rel(algo1_temp,algo2_temp)
	#print(t_test)

	t_test_value = t_test[0]

	if t_test[1] > 0.05:
		t_test_value = 0

	index = int(algo1[i][0])

	#if t test result is positive, algo1 is better, if t test result is negative, algo2 is better, if t test result is 0, then draw
	if t_test_value == 0:
		label[index] = 0
	elif t_test_value > 0:
		label[index] = 1
	elif t_test_value < 0:
		label[index] = 2

expertise_label_file = algo1_name + "_vs_" + algo2_name + ".txt"
with open(expertise_label_file,"w") as f:
	writer = csv.writer(f)
	for i in range(n_d):
		if label[i] != "NULL":
			a = []
			a.append(i)
			a.append(label[i])
			writer.writerow(a)

'''
	x = []
	x.append(algo1[i][0])
	x.append(t_test_value)
	x.append(t_test[1])

	with open("t_test_results.txt","a") as f:
		writer = csv.writer(f)
		writer.writerow(x)
'''


