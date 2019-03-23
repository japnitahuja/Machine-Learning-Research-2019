import numpy as np 
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

home = os.getcwd()
path = os.path.join(home, "scatter plot")
#if not os.path.isdir(path):
	#os.mkdir(path)

with open("complexity_cluster_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_complexity = [row for row in reader]
	cluster_wise_complexity = np.array(cluster_wise_complexity).astype(np.float64)

with open("cluster_wise_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	cluster_wise_accuracy = [row for row in reader]
	cluster_wise_accuracy = np.array(cluster_wise_accuracy).astype(np.float64)

with open("complexity_baseline_format_normal.txt", "r") as f:
	reader = csv.reader(f)
	baseline_complexity = [row for row in reader]
	baseline_complexity = np.array(baseline_complexity).astype(np.float64)

'''
with open("baseline_accuracy.txt", "r") as f:
	reader = csv.reader(f)
	baseline_accuracy = [row for row in reader]
	baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)
'''

with open("hybrid_overall_acc.txt", "r") as f:
	reader = csv.reader(f)
	line = [row for row in reader]

baseline_accuracy = line[2][:]
baseline_accuracy = np.array(baseline_accuracy).astype(np.float64)

#calculate difference between baseline complexity and cluster wise complexity
for feature in range(22):
	for n0 in range(100):
		for cluster in range(4):
			cluster_wise_complexity[n0*4+cluster][feature] -= baseline_complexity[n0][feature]

#calculate difference between baseline accuracy and cluster wise accuracy
for n0 in range(100):
	for cluster in range(4):
		cluster_wise_accuracy[n0*4+cluster][3] -= baseline_accuracy[n0]

plt.figure(figsize=(30,30))
for feature in range(22):
	print("feature", feature)
	x = []
	y = []
	for i in range(400):
		x.append(cluster_wise_complexity[i][feature])
		y.append(cluster_wise_accuracy[i][3])
	#print(len(x))
	#print(len(y))
	#plt.clf()
	plt.subplot(5,5,feature+1)
	plt.scatter(x, y)
	#fig = plt.figure()
	plot_title = "complexity " + str(feature)
	plt.title(plot_title)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
#plt.show()

#os.chdir(path)
plt_name = str(feature) + ".png"
plt.savefig(plt_name, dpi=300)
#os.chdir(home)
